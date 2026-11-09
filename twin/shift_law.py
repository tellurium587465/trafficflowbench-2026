"""Empirical law of the real scenario redraw at the junctions (validation month only; pooled over panels; only the
distribution shape leaves this script, never per-junction values): delta = median weekday-morning free-flow log ratio
(validation) - train mean, against the junction's size (|train mean log ratio|, the connector's net flow share) and
the within-train sd. Writes work/twin/shift_law.json: per size bin, the share of 'moving' junctions (|delta| > 3 sd)
and the quantiles of |delta| among them, plus the sd of the rest.
usage: python shift_law.py"""
import sys, json, numpy as np, pandas as pd, warnings
warnings.simplefilter("ignore")
sys.path.insert(0, r"C:/Users/user/tfb/src"); sys.path.insert(0, r"C:/Users/user/tfb/twin")
from cachelib import load, T2_PANELS
exec(open(r"C:/Users/user/tfb/twin/shift_diag.py", encoding="utf-8").read().split("rows = []")[0].split('"""', 2)[2])
rows = []
for p in T2_PANELS:
    Pt = load(p, "train"); months = sorted(set(str(x)[:7] for x in Pt.day[:Pt.T // 288]))
    M = np.array([jr(Pt, m)[0] for m in months]); mean = np.nanmean(M, axis=0); sd = np.nanstd(M, axis=0) + 1e-4
    x, _ = jr(load(p, "validation"))
    for j in range(len(mean)):
        if np.isfinite(x[j]) and np.isfinite(mean[j]):
            rows.append(dict(panel=p, size=abs(mean[j]), sd=sd[j], delta=x[j] - mean[j]))
R = pd.DataFrame(rows); R["z"] = R.delta / R.sd; R["mov"] = R.z.abs() > 3
bins = [0, 0.002, 0.01, 0.05, 10]; R["sb"] = pd.cut(R["size"], bins, include_lowest=True)
law = {}
for b, g in R.groupby("sb", observed=True):
    m = g[g.mov]
    law[str(b)] = dict(lo=float(b.left), hi=float(b.right), n=int(len(g)), share_moving=float(g.mov.mean()),
                       moving_abs_q=[float(np.quantile(m.delta.abs(), q)) for q in (0.1, 0.25, 0.5, 0.75, 0.9)] if len(m) else [0.0] * 5,
                       rest_sd=float(g[~g.mov].delta.std()), sign_pos=float((m.delta > 0).mean()) if len(m) else 0.5)
    print(b, f"n {len(g)} moving {g.mov.mean():.3f} |delta| of moving q50 {law[str(b)]['moving_abs_q'][2]:.4f} q90 {law[str(b)]['moving_abs_q'][4]:.4f} rest sd {law[str(b)]['rest_sd']:.4f}")
json.dump(law, open(r"C:/Users/user/tfb/work/twin/shift_law.json", "w"), indent=1)
# are moving junctions clustered along the corridor (one connector) or scattered?
cl = R.groupby("panel").apply(lambda g: np.mean(np.diff(np.flatnonzero(g.mov.to_numpy())) == 1) if g.mov.sum() > 1 else np.nan, include_groups=False)
print("share of moving junctions adjacent to another moving one, by panel:", cl.round(2).to_dict())
