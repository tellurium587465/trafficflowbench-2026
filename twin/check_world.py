"""Twin sanity check: congestion-map shift. Per month, the per-link share of queued eligible cells (normalised to sum
1); total-variation distance between months. Real train months pairwise (F1: <= 0.12), twinA months pairwise, twinA
vs each test world (target: the real test-vs-train range 0.10-0.40, F1). Also twinA vs the real train months."""
import sys, numpy as np, pandas as pd
sys.path.insert(0, r"C:/Users/user/tfb/src")
from cachelib import load
panel = sys.argv[1]; ntest = int(sys.argv[2]) if len(sys.argv) > 2 else 2


def maps(P):
    Q = (np.nan_to_num(P.V, nan=1e9) <= P.vcut[None, :]) & P.E
    mon = np.array([str(t)[:7] for t in P.times]); out = {}
    for m in np.unique(mon):
        f = Q[mon == m].sum(0).astype(float); out[m] = f / max(f.sum(), 1)
    return out


tv = lambda a, b: 0.5 * np.abs(a - b).sum()
R = maps(load(panel, "train")); A = maps(load(panel, "twinA"))
rr = [tv(R[a], R[b]) for i, a in enumerate(R) for b in list(R)[i + 1:]]
aa = [tv(A[a], A[b]) for i, a in enumerate(A) for b in list(A)[i + 1:]]
ra = [tv(R[m], A[m]) for m in R if m in A]
Am = np.mean(list(A.values()), axis=0)
tt = []
for k in range(ntest):
    T = maps(load(panel, f"twinT{k}")); tt += [tv(v, Am) for v in T.values()]
print(f"{panel}: real train months pairwise TV {np.mean(rr):.3f} (max {np.max(rr):.3f}) | twinA pairwise {np.mean(aa):.3f} (max {np.max(aa):.3f}) | "
      f"twinA vs real same month {np.mean(ra):.3f} | test worlds vs twinA mean {np.round(tt, 3).tolist()}")
