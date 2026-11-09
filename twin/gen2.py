"""Digital twin v2 (10/13): gen.py plus stochastic site capacities (fid_replay2.py: same-site fidelity 0.425 -> 0.608).
Identification uses the train months 0-5 only (2030-06..2030-11), so that months 6-8 stay a clean real test.
  twinB      : world A analogue, every day of months 0-5 replayed (real upstream demand), site capacity drawn per day
  twinQ<k>   : randomized worlds (template month k % 6, 30 days): gen.py's redraws (capacity / free speed per link,
               junction ratios, upstream demand) + per-site median capacity shifted by exp(N(0, SIG_SITE)) + per-day draws
Incidents, noise, masks and release fields as gen.py.
usage: python gen2.py <panel|ALL> [n_Q=16] [seed=7]"""
import sys, time, zlib, numpy as np, pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm
sys.path.insert(0, r"C:/Users/user/tfb/src"); sys.path.insert(0, r"C:/Users/user/tfb/twin")
from cachelib import load, T2_PANELS
from ctm import Corridor, junction_ratios
import gen
SIG_SITE = 0.03
JR_ROWS = None          # review 10/13: rows the junction ratios are estimated from (months 0-5 only), set by main()
E = pd.read_csv(r"C:/Users/user/tfb/work/t2/g34_events.csv")


def months_of(P):
    D = P.T // 288; return sorted(set(str(x)[:7] for x in P.day[:D]))


def ma_max(P, b, lo, hi):
    """max over rows lo..hi-1 of the 30-min moving average flow, max over links b..b+2 (NaN-aware)"""
    q = P.Q[max(0, lo - 5):hi, b:b + 3]
    if not np.isfinite(q).any():
        return np.nan
    c = np.nancumsum(np.nan_to_num(q), axis=0); n = np.cumsum(np.isfinite(q), axis=0)
    c = np.vstack([np.zeros((1, q.shape[1])), c]); n = np.vstack([np.zeros((1, q.shape[1])), n])
    k = 6; s = (c[k:] - c[:-k]) / np.maximum(n[k:] - n[:-k], 1); s[(n[k:] - n[:-k]) < 4] = np.nan
    return np.nanmax(s) if np.isfinite(s).any() else np.nan


def site_dists(P, keep_days):
    """censored lognormal capacity per regular site from the days in keep_days only. Review 10/13: the SAME statistic on
    both sides (it was a single 30-min mean before breakdown vs the max over 42 block means on survival mornings, which
    inflated sigma): breakdown morning -> max of the 30-min moving average (links b..b+2) from 05:00 up to the first
    queue row (the capacity, observed); morning without a queue at the site -> the same max over 05:00-12:00 (a lower
    bound, censored)."""
    qv = (P.V <= P.vcut[None, :]) & np.isfinite(P.V); out = {}
    Ep = E[(E.panel == P.panel) & E.day.isin(keep_days)]
    for b, g in Ep.groupby("b"):
        if len(g) < 3:
            continue
        brk = np.log(np.array([ma_max(P, b, d * 288 + 60, d * 288 + s_) for d, s_ in zip(g.day, g.slot)]))
        brk = brk[np.isfinite(brk)]; surv = []
        for d in keep_days:
            lo, hi = d * 288 + 60, d * 288 + 144
            if qv[lo:hi, max(0, b - 4):b].any():
                continue
            v = ma_max(P, b, lo, hi)
            if np.isfinite(v):
                surv.append(np.log(v))
        surv = np.array(surv)
        if len(brk) < 3:
            continue
        nll = lambda th: -(norm.logpdf(brk, th[0], np.exp(th[1])).sum() + norm.logsf(surv, th[0], np.exp(th[1])).sum())
        r = minimize(nll, [brk.mean(), np.log(max(brk.std(), 0.01))], method="Nelder-Mead")
        out[int(b)] = (float(r.x[0]), float(np.exp(r.x[1])))
    return out


def base_cap_mult(P, keep_days, dists):
    rows = np.concatenate([np.arange(d * 288, (d + 1) * 288) for d in keep_days])
    m30 = np.nanmean(P.Q[rows].reshape(-1, 6, P.n), axis=1)
    cm = np.maximum(P.cap, 1.02 * np.nanmax(m30, axis=0)) / P.cap
    for b in dists:
        cm[b:b + 3] = np.maximum(cm[b:b + 3], 1.0)
    return cm


def simulate2(PT, days, rng, cap_mult, vf_mult, rat_shift, up_mult, dists, site_shift):
    vf0 = PT.vf.copy(); PT.vf = PT.vf * vf_mult
    try:
        C = Corridor(PT, cap_mult=cap_mult)
    finally:
        PT.vf = vf0
    RAT = junction_ratios(PT, JR_ROWS) * np.exp(rat_shift)[None, None, :]
    dayt = np.array([pd.Timestamp(str(PT.day[d])).dayofweek >= 5 for d in days]).astype(int)
    CR = C.cell_ratios(RAT)[dayt]
    up = pd.Series(PT.Q[:, 0].astype(float)).interpolate(limit_direction="both").to_numpy()
    U = np.stack([up[d * 288:(d + 1) * 288] for d in days]) * up_mult
    capf, log = gen.incidents(rng, len(days), C.NC)
    for b, (mu, s) in dists.items():
        draw = np.exp(mu + site_shift.get(b, 0.0) + s * rng.standard_normal(len(days)))
        for c in np.unique(C.cell_of[b:b + 3]):
            capf[:, :, c] = np.minimum(capf[:, :, c], (draw / C.cap[c])[:, None])
    Vc, Fc = C.run(U, CR, capf)
    lr = np.log(RAT); same = C.cell_of[:-1] == C.cell_of[1:]
    G = np.zeros((2, 288, PT.n))
    for i in range(1, PT.n):
        G[:, :, i] = np.where(same[i - 1], G[:, :, i - 1] + lr[:, :, i - 1], 0.0)
    G = np.exp(G)[dayt]
    V = Vc[:, :, C.cell_of].reshape(-1, PT.n); Q = (Fc[:, :, C.cell_of] * G).reshape(-1, PT.n)
    return V.astype(np.float32), Q.astype(np.float32), log


def main(panel, nQ=16, seed=7):
    t0 = time.time(); PT = load(panel, "train"); D = PT.T // 288
    rng = np.random.default_rng([seed, zlib.crc32(panel.encode())])
    months = months_of(PT)[:6]
    keep = [d for d in range(D) if str(PT.day[d])[:7] in months]
    global JR_ROWS
    JR_ROWS = np.isin(np.arange(PT.T) // 288, keep)
    dists = site_dists(PT, keep); cm = base_cap_mult(PT, keep, dists)
    one = np.ones(PT.n); zero = np.zeros(PT.n - 1)
    V, Q, log = simulate2(PT, keep, rng, cm, one, zero, 1.0, dists, {})
    rows = np.concatenate([np.arange(d * 288, (d + 1) * 288) for d in keep])
    gen.write(panel, "twinB", PT, keep, V, Q, rng, list(PT.times[rows]), dict(world="B", incidents=log, days=len(keep), sites=dists))
    print(panel, "twinB", len(dists), "sites", f"{time.time()-t0:.0f}s", flush=True)
    base_lr = np.log(junction_ratios(PT, JR_ROWS)).mean(axis=(0, 1))
    for k in range(nQ):
        tm = months[k % len(months)]; days = [d for d in range(D) if str(PT.day[d]).startswith(tm)][:30]
        cap_mult = cm * np.exp(gen.SIG_CAP * rng.standard_normal(PT.n)); vf_mult = np.exp(gen.SIG_VF * rng.standard_normal(PT.n))
        rat = gen.SIG_RAT * np.abs(base_lr) * rng.standard_normal(PT.n - 1); upm = float(np.exp(gen.SIG_UP * rng.standard_normal()))
        shift = {b: float(SIG_SITE * rng.standard_normal()) for b in dists}
        V, Q, log = simulate2(PT, days, rng, cap_mult, vf_mult, rat, upm, dists, shift)
        y, m = gen.far_calendar(pd.Timestamp(str(PT.day[days[0]])).dayofweek, k + 160)
        base = pd.Timestamp(y, m, 1)
        times = [(base + pd.Timedelta(minutes=5 * i)).strftime("%Y-%m-%dT%H:%M:%SZ") for i in range(len(days) * 288)]
        gen.write(panel, f"twinQ{k}", PT, days, V, Q, rng, times, dict(world=f"Q{k}", template=tm, up_mult=upm, incidents=log, days=len(days)))
    print(panel, f"twinQ0..{nQ-1}", f"{time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    pans = T2_PANELS if sys.argv[1] == "ALL" else sys.argv[1].split(",")
    for p in pans:
        main(p, int(sys.argv[2]) if len(sys.argv) > 2 else 16, int(sys.argv[3]) if len(sys.argv) > 3 else 7)
