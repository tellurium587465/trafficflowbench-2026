"""B16 (10/29, strategy lane 10 (a)): is the training months' month-to-month turnover of congestion places comparable to the
training -> March change? If consecutive training months barely turn over, a rolling-origin instrument on them
under-states the test shift (as synthetic v4 did). Per panel and month pair A -> B (observed cells only): share of B's
congested cells at links whose congestion rate in A is < 1 % of rows ("new places"), and the rank correlation of the
per-link congestion rates. Pairs: consecutive training months, Feb -> March (validation), and all 9 training months -> March.
Diagnostic only (March aggregates, not used to tune anything)."""
import sys, numpy as np, pandas as pd
sys.path.insert(0, r"C:/Users/user/tfb/src")
from cachelib import load, T2_PANELS
from scipy.stats import spearmanr
rows = []
for p in T2_PANELS:
    P = load(p, "train"); ym = np.array([t[:7] for t in P.times]); M = sorted(set(ym))
    cong = (P.V <= P.vcut[None, :]) & np.isfinite(P.V); obs = np.isfinite(P.V)
    rate = {m: cong[ym == m].sum(0) / np.maximum(obs[ym == m].sum(0), 1) for m in M}; cnt = {m: cong[ym == m].sum(0) for m in M}
    Pv = load(p, "validation"); cv = (Pv.V <= Pv.vcut[None, :]) & np.isfinite(Pv.V)
    rate["2031-03"] = cv.sum(0) / np.maximum(np.isfinite(Pv.V).sum(0), 1); cnt["2031-03"] = cv.sum(0)
    rate["train"] = cong.sum(0) / np.maximum(obs.sum(0), 1)
    pairs = [(M[i], M[i + 1]) for i in range(len(M) - 1)] + [("2031-02", "2031-03"), ("train", "2031-03")]
    for a, b in pairs:
        new = rate[a] < 0.01; share = cnt[b][new].sum() / max(cnt[b].sum(), 1)
        rho = spearmanr(rate[a], rate[b]).correlation
        rows.append((p, a, b, share, rho))
    load.cache_clear()
D = pd.DataFrame(rows, columns=["panel", "A", "B", "new_share", "rho"])
D["kind"] = np.where(D.B == "2031-03", np.where(D.A == "train", "train->Mar", "Feb->Mar"), "consecutive training")
print(D.groupby("kind")[["new_share", "rho"]].agg(["median", "min", "max"]).round(3).to_string())
print(D[D.kind != "consecutive training"].pivot(index="panel", columns="kind", values="new_share").round(3).to_string())
c = D[D.kind == "consecutive training"].groupby("panel").new_share.median().rename("consec_median").round(3); print(c.to_string())
