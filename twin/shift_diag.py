"""Shift structure, real vs twin (aggregate magnitudes only, as the map TV was used before; no per-junction values of the
test months enter any model). Per junction, the median free-flow log ratio log(q_{i+1}/q_i) on weekday mornings
(06:00-10:00) per month. Reports, per panel:
  within  : sd across the 9 train months (month-to-month noise of one configuration)
  real    : |validation - train mean| and |private - train mean|   (the real scenario redraw)
  twin    : |T world - world A| (twin redraw, current gen.py) and |Q world - world B| (gen2.py)
as the median over junctions and the share of junctions moving by > 3 x the within sd; plus the upstream demand level
change (median morning flow at the first link).
usage: python shift_diag.py"""
import sys, glob, json, numpy as np, pandas as pd, warnings
warnings.simplefilter("ignore")
sys.path.insert(0, r"C:/Users/user/tfb/src")
from cachelib import load, T2_PANELS


def jr(P, month=None):
    D = P.T // 288; days = [d for d in range(D) if pd.Timestamp(str(P.day[d])).dayofweek < 5 and (month is None or str(P.day[d])[:7] == month)]
    rows = np.concatenate([np.arange(d * 288 + 72, d * 288 + 120) for d in days])
    V, Q = P.V[rows], P.Q[rows]
    ff = np.isfinite(V) & np.isfinite(Q) & (V > 0.85 * P.vf[None, :]) & (Q > 100)
    ok = ff[:, :-1] & ff[:, 1:]
    lr = np.where(ok, np.log(np.maximum(Q[:, 1:], 1) / np.maximum(Q[:, :-1], 1)), np.nan)
    return np.nanmedian(lr, axis=0), np.nanmedian(Q[:, 0])


rows = []
for p in T2_PANELS:
    Pt = load(p, "train"); months = sorted(set(str(x)[:7] for x in Pt.day[:Pt.T // 288]))
    M = np.array([jr(Pt, m)[0] for m in months]); up = np.array([jr(Pt, m)[1] for m in months])
    mean = np.nanmean(M, axis=0); sd = np.nanstd(M, axis=0) + 1e-4; upm = np.nanmean(up)
    def stats(name, x, u):
        d = np.abs(x - mean); return dict(panel=p, kind=name, med_abs=np.nanmedian(d), share_3sd=np.nanmean(d > 3 * sd), up_change=abs(np.log(u / upm)))
    rows.append(dict(panel=p, kind="within", med_abs=np.nanmedian(sd), share_3sd=np.nan, up_change=np.nanstd(np.log(up / upm))))
    for sp in ("validation", "private"):
        x, u = jr(load(p, sp)); rows.append(stats(sp, x, u))
    xa, ua = jr(load(p, "twinA"))
    for f in sorted(glob.glob(f"C:/Users/user/tfb/work/cache/{p}_twinT[0-3].npz")):
        w = f.split("_")[-1][:-4]; x, u = jr(load(p, w))
        d = np.abs(x - xa); rows.append(dict(panel=p, kind="twin T-A", med_abs=np.nanmedian(d), share_3sd=np.nanmean(d > 3 * sd), up_change=abs(np.log(u / ua))))
    xb, ub = jr(load(p, "twinB"))
    for k in range(4):
        x, u = jr(load(p, f"twinQ{k}")); d = np.abs(x - xb)
        rows.append(dict(panel=p, kind="twin2 Q-B", med_abs=np.nanmedian(d), share_3sd=np.nanmean(d > 3 * sd), up_change=abs(np.log(u / ub))))
R = pd.DataFrame(rows)
print(R.groupby(["kind"])[["med_abs", "share_3sd", "up_change"]].median().round(4).to_string())
print(R.pivot_table(index="panel", columns="kind", values="med_abs", aggfunc="median").round(4).to_string())
