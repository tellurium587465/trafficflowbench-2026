"""Breakdown-type bottleneck in the CTM twin (10/14, A4). The real generator's congestion: a cliff onset (v/vcut 1.65 ->
congested level in one row), flow inside the queue ~0.74 x capacity, site-specific discrete congested speeds, slow growth,
random starts (hazard 3-10% per row). The twin so far (gen2: a per-day capacity draw held all day) lets demand creep over
the capacity, so its queues build up smoothly. Hypothesis: the generator uses a two-state bottleneck - when the flow at a
site reaches a random capacity, the site breaks down and discharges at alpha x capacity until its queue clears; the
congested level is then fixed by the congested branch of the FD at that discharge flow (one speed per site).
Modes (same identification as onset_shape2: months 0-5, gen2 site capacity distributions, junction ratios, real
upstream flow):
  daily      - gen2 as is (capacity factor draw/cap on the site cells all day)
  brk<alpha> - the site passes everything (base capacity) until its 30-min mean flow (max over links b..b+2, the same
               statistic the distributions were fitted on) reaches a per-day draw; then it discharges at alpha x the FD
               capacity of link b until the cell upstream is below critical density (>= 3 rows), then a new draw.
Compared with the real data on the same days: onset profile, first-row extent, two-run share, episode level / spread /
growth, and the morning head-site distribution (total variation distance). V/Q/E come from the raw C: data (the caches
are on the E: HDD the RSNA job reads); only small static arrays are read from the cache.
  epi<alpha> - episodes: start each row with a per-site logistic hazard of the 30-min flow / capacity (fitted on the
               real months 0-5), last a duration drawn from the panel's real morning episodes (episodes_raw.csv) and
               discharge at alpha x capacity meanwhile.
Modes may end in "@blk": the CTM cells are then the generator's blocks estimated on months 0-5 (block_structure.py)
instead of links merged to >= 0.12 km.
usage: python onset_shape3.py [panels=D12_I5_S,D7_I10_E] [modes=daily,brk0.74,brk0.85]"""
import sys, glob, numpy as np, pandas as pd, warnings
from scipy import ndimage
warnings.simplefilter("ignore")
sys.path.insert(0, r"C:/Users/user/tfb/src"); sys.path.insert(0, r"C:/Users/user/tfb/twin")
from cachelib import load
from common import REL
from ctm import Corridor, junction_ratios
import gen2


def raw_matrices(P):
    li = {l: i for i, l in enumerate(P.links)}; ti = {t: i for i, t in enumerate(P.times)}
    V = np.full((len(P.times), P.n), np.nan, np.float32); Q = V.copy(); E = np.zeros((len(P.times), P.n), bool)
    for f in sorted(glob.glob(str(REL / "corridors" / P.panel / "train" / "mainline_states" / "**" / "*.parquet"), recursive=True)):
        d = pd.read_parquet(f, columns=["timestamp", "link_id", "speed_kmh", "flow_vph", "is_score_eligible"])
        g = d.groupby(["timestamp", "link_id"], sort=False).agg(v=("speed_kmh", "mean"), q=("flow_vph", "mean"), e=("is_score_eligible", "max")).reset_index()
        r = g.timestamp.map(ti).to_numpy(); c = g.link_id.map(li).to_numpy(); ok = ~(pd.isna(r) | pd.isna(c))
        r, c = r[ok].astype(int), c[ok].astype(int)
        V[r, c] = g.v.to_numpy()[ok]; Q[r, c] = g.q.to_numpy()[ok]; E[r, c] = g.e.to_numpy()[ok] > 0
    return V, Q, E


def make_hook(P, C, dists, alpha, rng, D):
    sites = sorted(dists); S = len(sites)
    cs = [np.unique(C.cell_of[b:b + 3]) for b in sites]; cu = [max(C.cell_of[b] - 1, 0) for b in sites]
    kcrit = C.cap / C.vf
    mu = np.array([dists[b][0] for b in sites]); sg = np.array([dists[b][1] for b in sites])
    thr = np.exp(mu[None, :] + sg[None, :] * rng.standard_normal((D, S)))
    broken = np.zeros((D, S), bool); since = np.zeros((D, S), int); log = []

    def hook(r, V, F, k):
        nonlocal thr
        if r >= 6:
            for j in range(S):
                ma = F[:, r - 6:r][:, :, cs[j]].mean(axis=1).max(axis=-1)
                rec = broken[:, j] & (r - since[:, j] >= 3) & (k[:, cu[j]] < kcrit[cu[j]])
                if rec.any():
                    broken[rec, j] = False
                    thr[rec, j] = np.exp(mu[j] + sg[j] * rng.standard_normal(int(rec.sum())))
                trig = ~broken[:, j] & (ma >= thr[:, j])
                if trig.any():
                    broken[trig, j] = True; since[trig, j] = r
                    log.extend((int(d), r, sites[j]) for d in np.flatnonzero(trig))
        fac = np.ones((D, C.NC))
        for j, b in enumerate(sites):
            if broken[:, j].any():
                lim = np.minimum(alpha * P.cap[b] / C.cap[cs[j]], 1.0)
                fac[np.ix_(broken[:, j], cs[j])] = np.minimum(fac[np.ix_(broken[:, j], cs[j])], lim[None, :])
        return fac
    return hook, log


def blocks_of(V, P, rows):
    """the generator's blocks (block_structure.py) from the given rows: adjacent pairs with split rate <= 0.05."""
    ok = np.isfinite(V[rows]); cong = (V[rows] <= P.vcut[None, :]) & ok
    a, b = cong[:, :-1], cong[:, 1:]; both = ok[:, :-1] & ok[:, 1:]
    n_any = ((a | b) & both).sum(0); split = ((a ^ b) & both).sum(0) / np.maximum(n_any, 1)
    tight = (n_any >= 30) & (split <= 0.05); bid = np.zeros(P.n, int)
    for i in range(1, P.n):
        bid[i] = bid[i - 1] if tight[i - 1] else bid[i - 1] + 1
    return bid


def ma30(Q, cols):
    """30-min moving average (rows t-5..t) of flow, max over the given links, NaN-aware -> (rows,)"""
    q = Q[:, cols]; c = np.vstack([np.zeros((1, q.shape[1])), np.nancumsum(np.nan_to_num(q), axis=0)])
    n = np.vstack([np.zeros((1, q.shape[1])), np.cumsum(np.isfinite(q), axis=0)])
    m = (c[6:] - c[:-6]) / np.maximum(n[6:] - n[:-6], 1); m[(n[6:] - n[:-6]) < 4] = np.nan
    return np.nanmax(np.vstack([np.full((5, q.shape[1]), np.nan), m]), axis=1)


def site_hazards(P, keep, sites):
    """per site b: logistic hazard of an episode start in row t given x = 30-min flow (links b..b+2, row t-1) / cap_b,
    fitted on the real at-risk morning rows (05:00-12:00, before the day's first start at b, no queue at b-4..b-1) of the
    identification days only; starts = gen2's breakdown events (g34)."""
    qv = (P.V <= P.vcut[None, :]) & np.isfinite(P.V); out = {}
    for b in sites:
        x = ma30(P.Q, list(range(b, min(b + 3, P.n)))) / P.cap[b]
        ev = gen2.E[(gen2.E.panel == P.panel) & (gen2.E.b == b)]; ev = {(int(d), int(s_)) for d, s_ in zip(ev.day, ev.slot)}
        X, Y = [], []
        for d in keep:
            for s_ in range(60, 144):
                t = d * 288 + s_; hit = (d, s_) in ev            # the start row already shows the queue: test it first
                if not hit and qv[t, max(0, b - 4):b].any():
                    break
                if np.isfinite(x[t - 1]):
                    X.append(x[t - 1]); Y.append(hit)
                if hit:
                    break
        X, Y = np.array(X), np.array(Y, float)
        print(f"   site {b}: at-risk rows {len(X)}, starts {int(Y.sum())}", flush=True); w = np.array([np.log(max(Y.mean(), 1e-4) / max(1 - Y.mean(), 1e-4)), 0.0])
        A = np.column_stack([np.ones(len(X)), X])
        for _ in range(25):                                      # IRLS, ridge 1e-3
            pr = 1 / (1 + np.exp(-A @ w)); W = pr * (1 - pr) + 1e-9
            w = w + np.linalg.solve(A.T @ (A * W[:, None]) + 1e-3 * np.eye(2), A.T @ (Y - pr) - 1e-3 * w)
        out[b] = w
    return out


def make_epi_hook(P, C, haz, durs, alpha, rng, D):
    """episodes: start ~ Bernoulli(hazard(x)) each row while the site is free, last a duration drawn from the real
    morning episodes of the panel, and cap the site's cells at alpha x cap_b meanwhile (then recover)."""
    sites = sorted(haz); S = len(sites); cs = [np.unique(C.cell_of[b:b + 3]) for b in sites]
    left = np.zeros((D, S), int); log = []

    def hook(r, V, F, k):
        if r >= 6:
            for j, b in enumerate(sites):
                x = F[:, r - 6:r][:, :, cs[j]].mean(axis=1).max(axis=-1) / P.cap[b]
                h = 1 / (1 + np.exp(-(haz[b][0] + haz[b][1] * x)))
                free = left[:, j] == 0; st = free & (rng.random(D) < h)
                left[~free, j] -= 1
                if st.any():
                    left[st, j] = rng.choice(durs, int(st.sum())); log.extend((int(d), r, b) for d in np.flatnonzero(st))
        fac = np.ones((D, C.NC))
        for j, b in enumerate(sites):
            on = left[:, j] > 0
            if on.any():
                lim = np.minimum(alpha * P.cap[b] / C.cap[cs[j]], 1.0)
                fac[np.ix_(on, cs[j])] = np.minimum(fac[np.ix_(on, cs[j])], lim[None, :])
        return fac
    return hook, log


def replay(P, cm, dists, jr, mode, seed=0, cell_of=None, haz=None, durs=None):
    rng = np.random.default_rng(seed); C = Corridor(P, cap_mult=cm, cell_of=cell_of); D = P.T // 288
    RAT = junction_ratios(P, jr); dayt = np.array([pd.Timestamp(str(x)).dayofweek >= 5 for x in P.day[:D]]).astype(int)
    CR = C.cell_ratios(RAT[dayt])
    up = pd.Series(P.Q[:, 0].astype(float)).interpolate(limit_direction="both").to_numpy()[:D * 288].reshape(D, 288)
    capf, hook, log = np.ones((D, 288, C.NC)), None, []
    if mode == "daily":
        for b, (mu, s) in dists.items():
            draw = np.exp(mu + s * rng.standard_normal(D))
            for c in np.unique(C.cell_of[b:b + 3]):
                capf[:, :, c] = np.minimum(capf[:, :, c], (draw / C.cap[c])[:, None])
    elif mode.startswith("epi"):
        hook, log = make_epi_hook(P, C, haz, durs, float(mode[3:]), rng, D)
    else:
        hook, log = make_hook(P, C, dists, float(mode[3:]), rng, D)
    Vc, Fc = C.run(up, CR, capf, hook=hook)
    return Vc[:, :, C.cell_of].reshape(-1, P.n), len(log)


def stats(P, V, E, D):
    vr = V / P.vf[None, :]; R = V / P.vcut[None, :]
    cong = (V <= P.vcut[None, :]) & np.isfinite(V); truth = cong & E[:D * 288]
    prof, two, nl, heads = [], 0, [], []
    for d in range(D):
        for s in range(72, 144):
            r = d * 288 + s
            if truth[r].any():
                idx = np.flatnonzero(truth[r]); prof.append(np.nanmedian(R[r - 3:r + 3][:, idx], axis=1)); nl.append(len(idx))
                two += len(np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1)) >= 2
                heads.append(int(idx.max()))
                break
    lab, n = ndimage.label(cong, structure=np.ones((3, 3))); sl = ndimage.find_objects(lab)
    lev, spr, first, traj, dur = [], [], [], [], []
    for i, s_ in enumerate(sl, 1):
        m = lab[s_] == i
        if m.sum() < 6 or not (72 <= s_[0].start % 288 < 144):
            continue
        vals = vr[s_][m]; ext = m.sum(1)
        lev.append(np.nanmedian(vals)); spr.append(np.nanquantile(vals, 0.9) - np.nanquantile(vals, 0.1)); first.append(ext[0])
        dur.append(s_[0].stop - s_[0].start); traj.append(np.pad(ext[:13].astype(float), (0, max(0, 13 - len(ext))), constant_values=np.nan))
    tr = np.nanmedian(np.array(traj), axis=0) if traj else np.full(13, np.nan)
    lv = np.round(np.array(lev), 2)
    top = pd.Series(lv).value_counts(normalize=True).head(3).round(2).to_dict() if len(lv) else {}
    return dict(mornings=len(prof), profile=np.round(np.nanmedian(np.array(prof), axis=0), 2).tolist() if prof else [],
                first_links=round(float(np.mean(nl)), 2) if nl else np.nan, two_run=round(two / max(len(prof), 1), 2),
                episodes=len(lev), level_med=round(float(np.median(lev)), 3) if lev else np.nan,
                spread_med=round(float(np.median(spr)), 3) if spr else np.nan, first_ext_med=float(np.median(first)) if first else np.nan,
                dur_med=float(np.median(dur)) if dur else np.nan, traj=tr.round(1).tolist(), top_levels=top), pd.Series(heads).value_counts(normalize=True)


panels = (sys.argv[1] if len(sys.argv) > 1 else "D12_I5_S,D7_I10_E").split(",")
modes = (sys.argv[2] if len(sys.argv) > 2 else "daily,brk0.74,brk0.85").split(",")
rows = []
for p in panels:
    P = load(p, "train"); D = P.T // 288
    P.V, P.Q, P.E = raw_matrices(P)
    months = sorted(set(str(x)[:7] for x in P.day[:D])); keep = [d for d in range(D) if str(P.day[d])[:7] in months[:6]]
    jr = np.isin(np.arange(P.T) // 288, keep); dists = gen2.site_dists(P, keep); cm = gen2.base_cap_mult(P, keep, dists)
    Vr = P.V[:D * 288]; st, hr = stats(P, Vr, P.E, D); print(p, "real", st, "heads", hr.head(3).round(2).to_dict(), flush=True)
    rows.append(dict(panel=p, mode="real", **st))
    bid = blocks_of(P.V, P, jr)
    haz = site_hazards(P, keep, sorted(dists)) if any(m.startswith("epi") for m in modes) else None
    ER = pd.read_csv(r"C:/Users/user/tfb/work/twin_episodes_raw.csv")
    durs = ER[(ER.panel == p) & (ER.start >= 72) & (ER.start < 144)].dur.to_numpy(int)
    if haz:
        print(p, "hazard (a, slope) per site", {b: np.round(w, 2).tolist() for b, w in haz.items()}, flush=True)
    for md in modes:
        base, _, cells = md.partition("@")                    # "<mode>@blk" = cells are the generator's blocks
        Vt, nb = replay(P, cm, dists, jr, base, cell_of=bid if cells == "blk" else None, haz=haz, durs=durs)
        Vt = np.where(np.isfinite(Vr), Vt, np.nan)
        st, ht = stats(P, Vt, P.E, D)
        tv = 0.5 * float(hr.reindex(hr.index.union(ht.index), fill_value=0).sub(ht.reindex(hr.index.union(ht.index), fill_value=0)).abs().sum())
        print(p, md, st, "heads", ht.head(3).round(2).to_dict(), f"head_TV {tv:.2f} breakdowns {nb}", flush=True)
        rows.append(dict(panel=p, mode=md, head_tv=round(tv, 3), breakdowns=nb, **st))
    load.cache_clear(); del P
pd.DataFrame(rows).to_csv(r"C:/Users/user/tfb/work/twin_onset_shape3.csv", index=False)
