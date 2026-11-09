"""G101 (10/24, report Fig. 2): one breakdown, as the generator produces it. D7_I10_E training months: the link queued most
often; the mornings where it switches from free flow (no queued row in the previous 60 min) to queued; the median-like
episode's speed / free-flow speed and flow / capacity per lane-capacity, 60 min before to 60 min after the first queued row.
Also the per-episode drop in v/v_f across the switch row, to show it is one row. -> work/t2/g101_regime_switch.csv
usage: python g101_regime_switch.py"""
import sys
sys.path.insert(0, r"C:/Users/user/tfb/src")
import numpy as np, pandas as pd
from cachelib import load

P = load("D7_I10_E", "train"); obs = np.isfinite(P.V); q = (P.V <= P.vcut[None, :]) & obs
j = int(np.argmax(q.sum(0) / np.maximum(obs.sum(0), 1)))
eps = []
for t in range(12, P.T - 12):
    if q[t, j] and not q[t - 12:t, j].any() and obs[t - 12:t, j].all() and (t % 288) < 12 * 12:
        eps.append(t)
drops = [float(P.V[t - 1, j] / P.vf[j] - P.V[t, j] / P.vf[j]) for t in eps]
pick = eps[int(np.argsort(drops)[len(drops) // 2])]
rows = [dict(minute=5 * (r - pick), v_over_vf=float(P.V[r, j] / P.vf[j]), q_over_cap=float(P.Q[r, j] / P.cap[j]), queued=bool(q[r, j]))
        for r in range(pick - 12, pick + 13)]
pd.DataFrame(rows).to_csv(r"C:/Users/user/tfb/work/t2/g101_regime_switch.csv", index=False)
print(f"link {j} ({P.links[j]}), morning switches {len(eps)}, drop of v/v_f across the switch row: median {np.median(drops):.2f}, "
      f"10-90% {np.percentile(drops, 10):.2f}-{np.percentile(drops, 90):.2f}; example at {P.times[pick]}")
