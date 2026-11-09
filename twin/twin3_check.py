"""Twin v3 fidelity on the Task-2 shape (10/14, TWIN_V3_PLAN.md). The generator: blocks as cells, a per-day random
breakdown threshold per site (gen2's censored lognormal), after breakdown a FIXED site discharge C_s (measured: the
median flow of the congested queue cells just upstream of the site, months 0-5), recovery when the cell upstream clears;
speed reported from the FD branch (congested cell -> congested-branch speed at its flow; free -> free speed), white
multiplicative noise (speed 0.8%, flow 2.3%).
Scored against the real data on the same days (identification months 0-5; audit 10/14: scored on months 6-8 only, and
each run writes its own work/twin3_check_<time>.csv):
  onset label size - morning onsets (first congested eligible cell 06-12 with no congestion in the 60 min before): the
                     number of congested eligible cells in rows r0..r0+5 (the horizon of an onset window)
  site            - the block of the most downstream congested link at r0 (TV distance real vs twin)
  cliff           - median v/vcut of the onset links in rows r0-3..r0+2
  level           - median v/vf of congested cells; onset mornings per day
Variants: v2 (links merged to 0.12 km, gen2 daily capacity, mean recording) | blk (block cells, daily capacity, mean) |
blkB (block cells, daily capacity, branch recording) | v3 (block cells, breakdown + fixed C_s, branch recording).
usage: python twin3_check.py [panels=D12_I5_S,D7_I10_E] [variants=v2,blk,blkB,v3]"""
import sys, numpy as np, pandas as pd, warnings
warnings.simplefilter("ignore")
sys.path.insert(0, r"C:/Users/user/tfb/src"); sys.path.insert(0, r"C:/Users/user/tfb/twin")
from cachelib import load
from ctm import Corridor, junction_ratios
from onset_shape3_lib import raw_matrices, blocks
import gen2


class Corridor3(Corridor):
    def run3(self, up, CR, capf, branch, hook=None, Cd=None):
        """ctm.Corridor.run with FD-branch recording (branch=True) and a per-row capacity hook; 5-min means otherwise.
        Cd: optional (NC,) measured discharge flow per cell (NaN = unknown): a congested cell then records Cd and the
        congested-branch speed at Cd (the real data sit at one fixed level per block)."""
        nd = up.shape[0]; NC = self.NC; dt = self.dt; kc = self.cap / self.vf
        k = np.zeros((nd, NC)); qin = np.zeros(nd)
        V = np.zeros((nd, 288, NC)); F = np.zeros((nd, 288, NC))
        for r in range(288):
            dem = up[:, r]; crj = CR[:, r][:, :-1]
            cap = self.cap[None, :] * capf[:, r]
            if hook is not None:
                cap = cap * hook(r, V, F, k)
            fl_acc = np.zeros((nd, NC)); k_acc = np.zeros((nd, NC))
            for _ in range(self.sub):
                S = np.minimum(self.vf * k, cap); R = np.minimum(cap, self.w * np.maximum(self.kj - k, 0))
                f = np.minimum(S[:, :-1], R[:, 1:] / np.maximum(crj, 1e-3))
                inflow0 = np.minimum(dem + qin / dt, R[:, 0]); qin = np.maximum(qin + (dem - inflow0) * dt, 0)
                inn = np.concatenate([inflow0[:, None], crj * f], axis=1); out = np.concatenate([f, S[:, -1:]], axis=1)
                k = np.maximum(k + dt / self.cl * (inn - out), 0); fl_acc += out; k_acc += k
            fl, kk = fl_acc / self.sub, k_acc / self.sub
            if branch:
                cong = kk > kc[None, :] * 1.02
                if Cd is not None:
                    fl = np.where(cong & np.isfinite(Cd)[None, :], np.nan_to_num(Cd)[None, :], fl)
                V[:, r] = np.where(cong, np.minimum(fl / np.maximum(self.kj - fl / self.w, 1e-6), self.vf), self.vf)
            else:
                V[:, r] = np.where(kk > 1e-3, np.minimum(fl / np.maximum(kk, 1e-6), self.vf), self.vf)
            F[:, r] = fl
        return V, F


def site_discharge(P, keep, sites):
    rows = np.isin(np.arange(P.T) // 288, keep); out = {}
    for b in sites:
        cols = list(range(max(0, b - 3), b)); v = P.V[rows][:, cols]; q = P.Q[rows][:, cols]
        m = (v <= P.vcut[cols][None, :]) & np.isfinite(q)
        out[b] = float(np.median(q[m])) if m.sum() >= 30 else np.nan
    return out


def make_hook(P, C, dists, Cs, rng, D):
    sites = [b for b in sorted(dists) if np.isfinite(Cs.get(b, np.nan))]; S = len(sites)
    cs = [np.unique(C.cell_of[b:b + 3]) for b in sites]; cu = [max(C.cell_of[b] - 1, 0) for b in sites]
    kc = C.cap / C.vf; mu = np.array([dists[b][0] for b in sites]); sg = np.array([dists[b][1] for b in sites])
    thr = np.exp(mu[None, :] + sg[None, :] * rng.standard_normal((D, S))); broken = np.zeros((D, S), bool); since = np.zeros((D, S), int)

    def hook(r, V, F, k):
        if r >= 6:
            for j in range(S):
                ma = F[:, r - 6:r][:, :, cs[j]].mean(axis=1).max(axis=-1)
                rec = broken[:, j] & (r - since[:, j] >= 3) & (k[:, cu[j]] < kc[cu[j]])
                if rec.any():
                    broken[rec, j] = False; thr[rec, j] = np.exp(mu[j] + sg[j] * rng.standard_normal(int(rec.sum())))
                trig = ~broken[:, j] & (ma >= thr[:, j])
                broken[trig, j] = True; since[trig, j] = r
        fac = np.ones((D, C.NC))
        for j, b in enumerate(sites):
            if broken[:, j].any():
                lim = np.minimum(Cs[b] / C.cap[cs[j]], 1.0)
                fac[np.ix_(broken[:, j], cs[j])] = np.minimum(fac[np.ix_(broken[:, j], cs[j])], lim[None, :])
        return fac
    return hook


def block_sites(P, keep, bid):
    """v4 sites at block level (10/14): the head block h of the real morning onsets (months 0-5, >= 5 onsets); the day's
    threshold = max 30-min block-mean flow OF THE CAPPED BLOCK h+1 (10/14: measured where the cap acts; h's own flow put
    the onsets at the wrong sites when a ramp sits between h and h+1) from 05:00 to the onset row (breakdown days) or to
    12:00 (censored); censored lognormal as gen2.site_dists. The bottleneck cell is the next block downstream (h+1), so
    the queue forms in h."""
    from scipy.stats import norm
    from scipy.optimize import minimize
    cg = (P.V <= P.vcut[None, :]) & np.isfinite(P.V) & P.E; nb = bid.max() + 1
    fb = np.stack([np.nanmean(P.Q[:, bid == j], axis=1) for j in range(nb)], axis=1)
    c = np.vstack([np.zeros((1, nb)), np.nancumsum(fb, axis=0)]); n = np.vstack([np.zeros((1, nb)), np.cumsum(np.isfinite(fb), axis=0)])
    ma = (c[6:] - c[:-6]) / np.maximum(n[6:] - n[:-6], 1); ma[(n[6:] - n[:-6]) < 4] = np.nan; ma = np.vstack([np.full((5, nb), np.nan), ma])
    first = {}
    for d in keep:
        for s_ in range(72, 144):
            r = d * 288 + s_
            if cg[r].any():
                if not cg[r - 12:r].any():
                    first[d] = (r, int(bid[np.flatnonzero(cg[r]).max()]))
                break
    cnt = pd.Series([h for _, h in first.values()]).value_counts(); out = {}
    for h in [int(x) for x in cnt.index if cnt[x] >= 5 and x + 1 < nb]:
        brk, surv = [], []
        for d in keep:
            lo = d * 288 + 60
            if d in first and first[d][1] == h:
                v = np.nanmax(ma[lo:first[d][0], h + 1]); brk.append(np.log(v)) if np.isfinite(v) else None
            elif d not in first:
                v = np.nanmax(ma[lo:d * 288 + 144, h + 1]); surv.append(np.log(v)) if np.isfinite(v) else None
        brk, surv = np.array(brk), np.array(surv)
        nll = lambda th: -(norm.logpdf(brk, th[0], np.exp(th[1])).sum() + norm.logsf(surv, th[0], np.exp(th[1])).sum())
        r_ = minimize(nll, [brk.mean(), np.log(max(brk.std(), 0.01))], method="Nelder-Mead")
        out[h] = (float(r_.x[0]), float(np.exp(r_.x[1])))
    return out


def simulate(P, variant, keep, bid, dists, Cs, seed=0, bsites=None):
    rng = np.random.default_rng(seed); D = P.T // 288; jr = np.isin(np.arange(P.T) // 288, keep)
    cm = gen2.base_cap_mult(P, keep, dists)
    C = Corridor3(P, cap_mult=cm, cell_of=None if variant == "v2" else bid)
    RAT = junction_ratios(P, jr); dayt = np.array([pd.Timestamp(str(x)).dayofweek >= 5 for x in P.day[:D]]).astype(int)
    CR = C.cell_ratios(RAT[dayt])
    up = pd.Series(P.Q[:, 0].astype(float)).interpolate(limit_direction="both").to_numpy()[:D * 288].reshape(D, 288)
    capf = np.ones((D, 288, C.NC)); hook = None
    if variant == "v3":
        hook = make_hook(P, C, dists, Cs, rng, D)
    elif variant == "v4":                                      # block sites: the day's threshold caps the block below h
        cellb = {j: int(C.cell_of[np.flatnonzero(bid == j)[0]]) for j in range(bid.max() + 1)}
        for h, (m_, s_) in bsites.items():
            draw = np.exp(m_ + s_ * rng.standard_normal(D)); c = cellb[h + 1]
            capf[:, :, c] = np.minimum(capf[:, :, c], (draw / C.cap[c])[:, None])
    else:
        for b, (m_, s_) in dists.items():
            draw = np.exp(m_ + s_ * rng.standard_normal(D))
            for c in np.unique(C.cell_of[b:b + 3]):
                capf[:, :, c] = np.minimum(capf[:, :, c], (draw / C.cap[c])[:, None])
    Cd = None
    if variant == "v4":                                        # measured discharge per block cell (months 0-5)
        rows = np.isin(np.arange(P.T) // 288, keep); cg = (P.V[rows] <= P.vcut[None, :]) & np.isfinite(P.Q[rows])
        Cd = np.full(C.NC, np.nan)
        for j in range(C.NC):
            m = cg[:, C.cell_of == j]
            if m.sum() >= 30:
                Cd[j] = np.median(P.Q[rows][:, C.cell_of == j][m])
    Vc, Fc = C.run3(up, CR, capf, branch=variant in ("blkB", "v3", "v4"), hook=hook, Cd=Cd)
    V = Vc[:, :, C.cell_of].reshape(-1, P.n); Q = Fc[:, :, C.cell_of].reshape(-1, P.n)
    V = V * (1 + 0.008 * rng.standard_normal(V.shape)); Q = Q * (1 + 0.023 * rng.standard_normal(Q.shape))
    return V, Q


def shape(P, V, E, bid, D, days=None):
    cg = (V <= P.vcut[None, :]) & np.isfinite(V) & E[:D * 288]; R = V / P.vcut[None, :]
    sizes, heads, prof, lev = [], [], [], []
    days = range(D) if days is None else days
    rows = np.zeros(D * 288, bool)
    for d in days:
        rows[d * 288:(d + 1) * 288] = True
    for d in days:
        for s in range(72, 144):
            r = d * 288 + s
            if cg[r].any():
                if cg[r - 12:r].any():
                    break                                      # queue already in the hour before: not an onset morning
                sizes.append(int(cg[r:r + 6].sum())); idx = np.flatnonzero(cg[r]); heads.append(int(bid[idx.max()]))
                prof.append(np.nanmedian(R[r - 3:r + 3][:, idx], axis=1))
                break
    vr = (V / P.vf[None, :])[cg & rows[:, None]]
    return dict(onset_mornings=len(sizes), per_day=round(len(sizes) / len(days), 3),
                size_q25_50_75=np.percentile(sizes, [25, 50, 75]).round(1).tolist() if sizes else [],
                cliff=np.round(np.nanmedian(np.array(prof), axis=0), 2).tolist() if prof else [],
                level=round(float(np.median(vr)), 3) if len(vr) else np.nan), pd.Series(heads).value_counts(normalize=True)


def main():
    panels = (sys.argv[1] if len(sys.argv) > 1 else "D12_I5_S,D7_I10_E").split(",")
    variants = (sys.argv[2] if len(sys.argv) > 2 else "v2,blk,blkB,v3").split(",")
    rows = []
    for p in panels:
        P = load(p, "train"); D = P.T // 288; P.V, P.Q, P.E = raw_matrices(P)
        months = sorted(set(str(x)[:7] for x in P.day[:D])); keep = [d for d in range(D) if str(P.day[d])[:7] in months[:6]]
        jr = np.isin(np.arange(P.T) // 288, keep); bid = blocks(P.V, P, jr)
        dists = gen2.site_dists(P, keep); Cs = site_discharge(P, keep, sorted(dists)); bs = block_sites(P, keep, bid)
        print(p, "block sites (head block: threshold, sd)", {h: (round(float(np.exp(m_))), round(s_, 3)) for h, (m_, s_) in bs.items()}, flush=True)
        print(p, "sites", {b: (round(float(np.exp(dists[b][0]))), round(Cs[b]) if np.isfinite(Cs[b]) else None) for b in sorted(dists)}, flush=True)
        test = [d for d in range(D) if d not in set(keep)]          # audit 10/14: score on months 6-8 only
        st, hr = shape(P, P.V[:D * 288], P.E, bid, D, test); print(p, "real", st, "heads", hr.head(3).round(2).to_dict(), flush=True)
        rows.append(dict(panel=p, variant="real", **st))
        for v in variants:
            Vt, Qt = simulate(P, v, keep, bid, dists, Cs, bsites=bs)
            Vt = np.where(np.isfinite(P.V[:D * 288]), Vt, np.nan)
            st, ht = shape(P, Vt, P.E, bid, D, test)
            ix = hr.index.union(ht.index); tv = 0.5 * float((hr.reindex(ix, fill_value=0) - ht.reindex(ix, fill_value=0)).abs().sum())
            print(p, v, st, "heads", ht.head(3).round(2).to_dict(), f"head_TV {tv:.2f}", flush=True)
            rows.append(dict(panel=p, variant=v, head_tv=round(tv, 3), **st))
        load.cache_clear(); del P
    import time
    pd.DataFrame(rows).to_csv(time.strftime(r"C:/Users/user/tfb/work/twin3_check_%Y%m%d_%H%M.csv"), index=False)


if __name__ == "__main__":
    main()
