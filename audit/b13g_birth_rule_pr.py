"""B13g (10/29, lane 11 success line (ii), strategy): can a jam's birth be written as a deterministic rule on the
denoised <= T state? Training months, the 57 known sites. Per site, rows where the site is free and the time of day is
inside the site's 5-95% birth band; state = mean flow per lane over the site's rectangle links, averaged over the last 3
rows (observed cells only), and its 3-row change; label = a birth at the site within the next 6 rows.
Rule = flow >= theta (and, second rule, flow >= theta and change >= 0). Per site: the best precision with recall >= 0.5.
Line (strategy, 10/29): a site with precision >= 0.95 and recall >= 0.5 -> a rule override worth measuring on M3."""
import sys, numpy as np, pandas as pd
sys.path.insert(0, r"C:/Users/user/tfb/src")
from cachelib import load
from common import WORK
from sklearn.metrics import precision_recall_curve
D = pd.read_parquet(WORK + "/b13_events.parquet"); S = pd.read_parquet(WORK + "/b13_sites.parquet")
out = []
for p, ss in S.groupby("panel"):
    P = load(p, "train"); ql = P.Q / P.lanes[None, :]; tod = np.arange(P.T) % 288
    for s in ss.itertuples():
        e = D[(D.panel == p) & (D["head"] == s.head)]; lo, hi = np.percentile(e.tod, [5, 95])
        jam = np.zeros(P.T, bool); birth = np.zeros(P.T, bool)
        for r in e.itertuples():
            jam[r.t0:r.t0 + r.dur] = True; birth[r.t0] = True
        f = np.nanmean(ql[:, s.tail:s.head + 1], axis=1)
        f3 = pd.Series(f).rolling(3, min_periods=1).mean().to_numpy(); d3 = f3 - np.r_[np.full(3, np.nan), f3[:-3]]
        soon = np.array([birth[t + 1:t + 7].any() for t in range(P.T)])
        ok = ~jam & (tod >= lo) & (tod <= hi) & np.isfinite(f3) & np.isfinite(d3)
        y = soon[ok]
        if y.sum() < 20:
            continue
        best = []
        for sc in (f3[ok], np.where(d3[ok] >= 0, f3[ok], -1e9)):
            pr, rc, _ = precision_recall_curve(y, sc); best.append(pr[rc >= 0.5].max())
        out.append((p, s.head, int(y.sum()), y.mean(), *best))
    load.cache_clear()
R = pd.DataFrame(out, columns=["panel", "head", "births", "base_rate", "prec_flow", "prec_flow_rising"])
print(R.sort_values("prec_flow", ascending=False).head(10).round(3).to_string())
print(f"sites {len(R)} | base rate (median) {R.base_rate.median():.3f} | best precision at recall >= 0.5 (median): flow {R.prec_flow.median():.3f}, flow & rising {R.prec_flow_rising.median():.3f} | max {R[['prec_flow', 'prec_flow_rising']].max().max():.3f}")
print("sites meeting precision >= 0.95 and recall >= 0.5:", int((R[["prec_flow", "prec_flow_rising"]].max(axis=1) >= 0.95).sum()))
