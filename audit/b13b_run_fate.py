"""B13b (10/29, lane 11 orientation): fate of a congested run h rows later in the training-month truth (B13's fill).
For every run at row t (every 3rd row), share of its links still congested at t+h (h = 1, 3, 6) and the contiguous upstream
extension beyond its tail. If jams are rectangles, the share is 0 or 1."""
import sys, numpy as np
sys.path.insert(0, r"C:/Users/user/tfb/src"); sys.path.insert(0, r"C:/Users/user/tfb/audit")
from cachelib import load
from common import PANELS
from b13_jam_shapes import fill, runs
H = (1, 3, 6); acc = {h: np.zeros(6) for h in H}; ext = {h: [] for h in H}
for p in PANELS:
    P = load(p, "train"); C = np.where(np.isnan(P.V), np.nan, (P.V <= P.vcut[None, :]).astype(float)); B = fill(C)
    for t in range(0, P.T - 6, 3):
        for a, b in runs(B[t]):
            for h in H:
                f = B[t + h, a:b + 1].mean(); e = 0
                while a - 1 - e >= 0 and B[t + h, a - 1 - e]:
                    e += 1
                if f == 0: e = 0
                k = 0 if f == 0 else (1 if f < 0.5 else (2 if f < 1 else 3))
                acc[h][k] += 1; acc[h][4] += (b - a + 1); acc[h][5] += (b - a + 1) * f
                ext[h].append(e)
    load.cache_clear()
for h in H:
    a = acc[h]; n = a[:4].sum(); e = np.array(ext[h])
    print(f"h={h}: runs {int(n)} | all gone {a[0]/n:.3f} | <50% left {a[1]/n:.3f} | 50-99% left {a[2]/n:.3f} | all left {a[3]/n:.3f} | upstream extension >0 links {np.mean(e > 0):.3f} (mean {e.mean():.2f})")
