"""Audit A12: which way does traffic flow in our link order?  The released topology says that in the S/W panels every
outgoing edge points to a LOWER index.  Empirical tests on train (unmasked):
 (1) free-flow lead-lag of flow differences between links ~8-15 km apart: corr(dq_i(t), dq_j(t+1)) vs corr(dq_i(t+1), dq_j(t))
 (2) queue growth: 30 min after the first queue of a day, does the queue extend to lower or higher index than the onset link?"""
import sys, numpy as np, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, r"C:/Users/user/tfb/src")
from cachelib import load
from common import PANELS

for p in PANELS:
    P = load(p, "train")
    x = np.concatenate([[0.0], np.cumsum(P.length[:-1])])
    V = np.where(P.E, P.V, np.nan)[: 288 * 120]; Q = np.where(P.E, P.Q, np.nan)[: 288 * 120]
    dq = np.diff(Q, axis=0)
    fw = bw = 0.0; npair = 0
    for i in range(0, P.n, 3):
        js = np.flatnonzero((x - x[i] > 8) & (x - x[i] < 15))
        if not len(js):
            continue
        j = js[0]
        a, b = dq[:-1, i], dq[1:, j]; c, d = dq[1:, i], dq[:-1, j]
        ok1 = np.isfinite(a) & np.isfinite(b); ok2 = np.isfinite(c) & np.isfinite(d)
        if ok1.sum() > 1000 and ok2.sum() > 1000:
            fw += np.corrcoef(a[ok1], b[ok1])[0, 1]; bw += np.corrcoef(c[ok2], d[ok2])[0, 1]; npair += 1
    q = (np.nan_to_num(V, nan=1e9) <= P.vcut[None, :])
    lo = hi = 0
    for d0 in range(0, q.shape[0] - 288, 288):
        s = q[d0:d0 + 288]; r = np.flatnonzero(s.any(axis=1))
        if len(r) == 0 or r[0] + 6 >= 288:
            continue
        on = np.flatnonzero(s[r[0]]); later = np.flatnonzero(s[r[0] + 6])
        if len(on) and len(later):
            lo += later.min() < on.min(); hi += later.max() > on.max()
    print(f"{p:11s} lead-lag corr low->high {fw/max(npair,1):+.3f} vs high->low {bw/max(npair,1):+.3f} ({npair} pairs) | "
          f"queue 30 min after onset extends to lower index on {lo} days, higher index on {hi} days")
    load.cache_clear()
