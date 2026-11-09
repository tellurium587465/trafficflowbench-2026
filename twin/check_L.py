"""Check the law-calibrated worlds (gen3) against the real redraw: junction shift magnitudes (as shift_diag.py) and the
congestion-map TV against the training configuration (real: validation / private vs the train-month mean map; twin:
T worlds vs world A, Q and L worlds vs world B; within: real train months pairwise)."""
import sys, numpy as np, pandas as pd, warnings
warnings.simplefilter("ignore")
sys.path.insert(0, r"C:/Users/user/tfb/src")
from cachelib import load, T2_PANELS
exec(open(r"C:/Users/user/tfb/twin/shift_diag.py", encoding="utf-8").read().split("rows = []")[0].split('"""', 2)[2])


def cmap(P, rows=None):
    """review 10/13: the same cells in every split (validation/private hide the Task 1 targets and the window blanks,
    the train months and the twins do not), so the maps are comparable"""
    m = P.E & np.isfinite(P.V) & ~P.target.astype(bool) & ~P.blank.astype(bool)
    if rows is not None:
        m = m & rows[:, None]
    Q = (P.V <= P.vcut[None, :]) & m
    f = Q.sum(0) / np.maximum(m.sum(0), 1)                       # queue share of the observed cells per link
    return f / max(f.sum(), 1e-9)


tv = lambda a, b: 0.5 * np.abs(a - b).sum()
rows = []
for p in T2_PANELS:
    Pt = load(p, "train"); months = sorted(set(str(x)[:7] for x in Pt.day[:Pt.T // 288]))
    M = np.array([jr(Pt, m)[0] for m in months]); mean = np.nanmean(M, axis=0); sd = np.nanstd(M, axis=0) + 1e-4
    tmap = cmap(Pt); A = cmap(load(p, "twinA")); B = cmap(load(p, "twinB")); xb, _ = jr(load(p, "twinB"))
    mon = np.array([str(x)[:7] for x in Pt.times])
    for mm in months:                                                     # within: one train month vs the others
        rows.append(dict(panel=p, kind="within", tv=tv(cmap(Pt, mon == mm), cmap(Pt, mon != mm))))
    for sp in ("validation", "private"):
        x, _ = jr(load(p, sp)); d = np.abs(x - mean)
        rows.append(dict(panel=p, kind=sp, jmed=np.nanmedian(d), share3=np.nanmean(d > 3 * sd), tv=tv(cmap(load(p, sp)), tmap)))
    for fam, ref, xr in (("T", A, None), ("Q", B, xb), ("L", B, xb)):
        for k in range(4):
            P = load(p, f"twin{fam}{k}"); row = dict(panel=p, kind=fam, tv=tv(cmap(P), ref))
            if xr is not None:
                x, _ = jr(P); d = np.abs(x - xr); row.update(jmed=np.nanmedian(d), share3=np.nanmean(d > 3 * sd))
            rows.append(row)
R = pd.DataFrame(rows)
print(R.groupby("kind")[["jmed", "share3", "tv"]].median().round(4).to_string())
print(R.pivot_table(index="panel", columns="kind", values="tv", aggfunc="median").round(3).to_string())
