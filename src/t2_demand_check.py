"""How predictable is the flow 35 min ahead (= demand that will meet a bottleneck at T+30)?
Compare on a held-out month, free-flow cells only (flow = demand there):
  persistence q(T-5) | linear trend over the last hour | profile ratio q(T-5) * P(dow, tod+35) / P(dow, tod)
where P is the per-link weekday time-of-day mean flow from the other months (generator: demand = weekday x ToD
profile x daily factor)."""
import numpy as np
from cachelib import load, T2_PANELS

STEP = 7                                              # T-5 -> T+30
res = []
for p in T2_PANELS:
    P = load(p, "train"); Q = np.where(P.E & (P.P >= 75), P.Q, np.nan); V = np.where(P.E & (P.P >= 75), P.V, np.nan)
    D = P.T // 288; Qd = Q[:D * 288].reshape(D, 288, P.n); Vd = V[:D * 288].reshape(D, 288, P.n)
    mon = np.array([d[:7] for d in P.day[:D]]); dow = np.array([np.datetime64(d).astype("datetime64[D]").view("int64") % 7 for d in P.day[:D]])
    for m in np.unique(mon)[-3:]:                    # last three months as hold-outs
        tr, te = mon != m, mon == m
        prof = np.full((7, 288, P.n), np.nan)
        for w in range(7):
            sel = tr & (dow == w)
            if sel.any():
                prof[w] = np.nanmean(Qd[sel], axis=0)
        for d in np.flatnonzero(te):
            q, v = Qd[d], Vd[d]; pr = prof[dow[d]]
            t = np.arange(12, 288 - STEP)
            y = q[t + STEP]; free = (v[t] > 0.9 * P.vf) & (v[t + STEP] > 0.9 * P.vf)
            per = q[t]
            tr_ = per + (np.nanmean(q[t[:, None] - np.arange(0, 3)[None, :]], axis=1) - np.nanmean(q[t[:, None] - np.arange(9, 12)[None, :]], axis=1)) * STEP / 9
            rat = per * pr[t + STEP] / np.maximum(pr[t], 1.0)
            ok = free & np.isfinite(y) & np.isfinite(per) & np.isfinite(tr_) & np.isfinite(rat)
            for name, est in (("persist", per), ("trend", tr_), ("profile", rat)):
                res.append((p, name, np.abs(est[ok] - y[ok]).sum(), np.abs(y[ok]).sum(), ok.sum()))
import pandas as pd
r = pd.DataFrame(res, columns=["p", "m", "ae", "y", "n"]).groupby(["p", "m"])[["ae", "y"]].sum()
r["rel_mae_%"] = 100 * r.ae / r.y
print(r["rel_mae_%"].unstack().round(2))
print("noise floor (flow obs noise ~2.3% -> MAE of noise diff ~", round(2.3 * np.sqrt(2) * 0.8, 2), "%)")
