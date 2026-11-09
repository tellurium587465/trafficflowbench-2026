"""G36: deterministic or stochastic capacity per site? For each frequent first-queue site s (G34 events, n >= 8):
breakdown flow bf on its own breakdown mornings vs the 'survived' flow on mornings with NO queue at s..s-3 all morning
(05:00-12:00): max over the morning of the 30-min mean flow at max(s..s+2). If survived flows often exceed the usual
breakdown flow, capacity is a per-day random draw (site choice among tied sites is then partly irreducible).
usage: python g36_capacity_stochastic.py"""
import sys, numpy as np, pandas as pd, warnings
warnings.simplefilter("ignore")
sys.path.insert(0, r"C:/Users/user/tfb/src")
from cachelib import load
sys.path.insert(0, r"C:/Users/user/tfb/twin")
from gen2 import ma_max      # review 10/13: the same 30-min moving-average max on both sides
E = pd.read_csv(r"C:/Users/user/tfb/work/t2/g34_events.csv")
out = []
for (p, s), g in E.groupby(["panel", "b"]):
    if len(g) < 8:
        continue
    P = load(p, "train"); D = P.T // 288
    qv = (P.V <= P.vcut[None, :]) & np.isfinite(P.V)
    bfs = np.array([ma_max(P, s, d * 288 + 60, d * 288 + sl) for d, sl in zip(g.day, g.slot)]); bfm = np.nanmedian(bfs); surv = []
    for d in range(D):
        lo, hi = d * 288 + 60, d * 288 + 144
        if qv[lo:hi, max(0, s - 4):s].any():                 # a queue at the site this morning (any time)
            continue
        surv.append(ma_max(P, s, lo, hi))
    surv = np.array(surv)
    out.append(dict(panel=p, site=s, n_break=len(g), bf_med=bfm, bf_p90=np.nanpercentile(bfs, 90), n_surv=len(surv),
                    surv_med=np.nanmedian(surv) if len(surv) else np.nan, share_surv_above_bfmed=np.nanmean(surv > bfm) if len(surv) else np.nan,
                    share_surv_above_bfp90=np.nanmean(surv > np.nanpercentile(bfs, 90)) if len(surv) else np.nan))
O = pd.DataFrame(out).sort_values("n_break", ascending=False)
print(O.round(3).to_string(index=False))
