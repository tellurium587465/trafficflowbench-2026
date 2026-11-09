"""Twin fidelity (IDEAS_TOP3 A4, audit B): real onset queues appear on 2-4 links at once (first queued row: 3.1 links, often
with 2-3 link gaps) while the twin's appear on ~1.2 links. Is the CTM cell length (lmin 0.12 km) the cause?
In memory only (no cache writes), 1-2 panels (small reads). Identification (site capacity distributions, junction ratios)
on the train months 0-5; scoring on the first morning queue of each day of months 6-8 (out of sample):
  real vs twin replay (real upstream demand, no incidents, site capacities drawn per day as gen2) for lmin in
  {0.12, 0.25, 0.5, 1.0} km: queued links at the first queued row (mean, share >= 3), number of runs, same-site rate.
usage: python cellsize_fidelity.py [panels=D12_I5_S,D7_I10_E]"""
import sys, numpy as np, pandas as pd, warnings
warnings.simplefilter("ignore")
sys.path.insert(0, r"C:/Users/user/tfb/src"); sys.path.insert(0, r"C:/Users/user/tfb/twin")
from cachelib import load
from ctm import Corridor, junction_ratios
import gen2


def first_rows(V, vcut, E, days):
    truth = (np.nan_to_num(V, nan=1e9) <= vcut[None, :]) & E; out = {}
    for d in days:
        for s in range(72, 144):
            r = d * 288 + s
            if truth[r].any():
                q = truth[r]; idx = np.flatnonzero(q)
                runs = 1 + int(np.sum(np.diff(idx) > 1)) if len(idx) else 0
                out[d] = (s, int(idx.max()), int(q.sum()), runs); break
    return out


def replay(P, lmin, cm, dists, jr_rows, seed=0, stochastic=True):
    rng = np.random.default_rng(seed); C = Corridor(P, lmin=lmin, cap_mult=cm); D = P.T // 288
    RAT = junction_ratios(P, jr_rows)
    dayt = np.array([pd.Timestamp(str(x)).dayofweek >= 5 for x in P.day[:D]]).astype(int)
    CR = C.cell_ratios(RAT[dayt])
    up = pd.Series(P.Q[:, 0].astype(float)).interpolate(limit_direction="both").to_numpy()[:D * 288].reshape(D, 288)
    capf = np.ones((D, 288, C.NC))
    for b, (mu, s) in dists.items():
        draw = np.exp(mu + s * rng.standard_normal(D)) if stochastic else np.full(D, np.exp(mu))
        for c in np.unique(C.cell_of[b:b + 3]):
            capf[:, :, c] = np.minimum(capf[:, :, c], (draw / C.cap[c])[:, None])
    Vc, Fc = C.run(up, CR, capf)
    return Vc[:, :, C.cell_of].reshape(-1, P.n)


pans = (sys.argv[1] if len(sys.argv) > 1 else "D12_I5_S,D7_I10_E").split(",")
rows = []
for p in pans:
    P = load(p, "train"); D = P.T // 288
    months = sorted(set(str(x)[:7] for x in P.day[:D]))
    keep = [d for d in range(D) if str(P.day[d])[:7] in months[:6]]; test = [d for d in range(D) if str(P.day[d])[:7] in months[6:]]
    jr_rows = np.isin(np.arange(P.T) // 288, keep)
    dists = gen2.site_dists(P, keep); cm = gen2.base_cap_mult(P, keep, dists)
    real = first_rows(P.V[:D * 288], P.vcut, P.E[:D * 288], test)
    def summ(name, fr):
        common = sorted(set(fr) & set(real))
        same = np.mean([abs(fr[d][1] - real[d][1]) <= 2 for d in common]) if common else np.nan
        a = np.array([v[2] for v in fr.values()]); r_ = np.array([v[3] for v in fr.values()])
        rows.append(dict(panel=p, variant=name, mornings=len(fr), links_mean=a.mean() if len(a) else np.nan,
                         share_ge3=np.mean(a >= 3) if len(a) else np.nan, runs_mean=r_.mean() if len(r_) else np.nan, same_site=same))
    summ("real", real)
    for lmin in (0.12, 0.25, 0.5, 1.0):
        V = replay(P, lmin, cm, dists, jr_rows)
        summ(f"twin lmin={lmin}", first_rows(V, P.vcut, P.E[:D * 288], test))
    load.cache_clear()
R = pd.DataFrame(rows); print(R.round(3).to_string(index=False))
