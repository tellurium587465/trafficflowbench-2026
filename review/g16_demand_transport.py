"""G16: does the demand profile (per-link weekday x time-of-day flow from the train months, t2_demand.py) transport to
the test months? Released observation layer only (a shift diagnostic like F1; nothing is fitted on it).
Forecast of flow 35 min ahead as in t2_demand: level = mean of q / prof over the last 3 rows; forecast = level x
prof(t+7). Scored on free-flow cells (finite, eligible, v > 0.85 vf at both ends) as relative MAE, days 1-4 only
(where the test windows are). Train months: leave-one-month-out profile; test months: all train months.
Baselines: persistence (q(t+7) = q(t-1)) and a link-free shape (corridor-mean profile ratio)."""
import sys, numpy as np, pandas as pd
sys.path.insert(0, r"C:/Users/user/tfb/src")
from cachelib import load, T2_PANELS
from t2_demand import profile, _dow

H = 7
res = []
for p in T2_PANELS:
    PT = load(p, "train")
    for sp in ("train", "validation", "private"):
        P = PT if sp == "train" else load(p, sp)
        D = P.T // 288; days = P.day[:D]
        sel = [d for d in range(D) if int(days[d][8:10]) <= 4]
        for d in sel:
            mon = days[d][:7]; pr = profile(PT, mon if sp == "train" else None)[_dow(days[d])]
            cm = np.nanmean(pr, axis=1)                                   # corridor-mean shape (no link identity)
            base = d * 288
            for s in range(84, 288 - H, 3):                               # 07:00 .. 23:25
                t = base + s
                rows = [t - 1, t - 2, t - 3]
                q0 = np.stack([P.Q[r] for r in rows]); v0 = np.stack([P.V[r] for r in rows])
                ok0 = np.isfinite(q0).all(0) & np.isfinite(v0).all(0) & (v0 > 0.85 * P.vf[None, :]).all(0)
                qt, vt = P.Q[t + H - 1], P.V[t + H - 1]
                ok = ok0 & np.isfinite(qt) & np.isfinite(vt) & (vt > 0.85 * P.vf) & P.E[t + H - 1] & (qt > 200)
                if ok.sum() < 3:
                    continue
                lv = np.mean([q0[i] / np.maximum(pr[(s - 1 - i) % 288], 1.0) for i in range(3)], axis=0)
                f_prof = lv * pr[(s + H - 1) % 288]
                f_pers = q0[0]
                f_cm = q0.mean(0) * cm[(s + H - 1) % 288] / max(np.mean([cm[(s - 1 - i) % 288] for i in range(3)]), 1.0)
                for nm, f in (("profile", f_prof), ("persist", f_pers), ("corridor_shape", f_cm)):
                    e = np.abs(f[ok] - qt[ok]) / qt[ok]
                    res.append((p, sp, nm, e.mean(), ok.sum()))
R = pd.DataFrame(res, columns=["panel", "split", "method", "rmae", "n"])
R["w"] = R.rmae * R.n
g = R.groupby(["split", "method"]).apply(lambda x: x.w.sum() / x.n.sum()).unstack()
print(g.round(4))
gp = R[R.method == "profile"].groupby(["panel", "split"]).apply(lambda x: x.w.sum() / x.n.sum()).unstack()
print(gp.round(4))
