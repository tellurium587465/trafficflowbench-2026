"""TE-A6: print v/vf maps (rows=time, cols=links) around isolated queue births. Usage: PANEL [k examples] [width]"""
import sys, numpy as np, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, r"C:/Users/user/tfb/src")
from cachelib import load

panel = sys.argv[1]; K = int(sys.argv[2]) if len(sys.argv) > 2 else 3; W = int(sys.argv[3]) if len(sys.argv) > 3 else 8
P = load(panel, "train")
V, Q = P.V, P.Q
C = np.isfinite(V) & (V <= P.vcut[None, :])
n, T = P.n, P.T
rng = np.random.default_rng(0)
cands = []
for t in range(12, T - 10):
    for j in np.flatnonzero(C[t] & ~C[t - 1]):
        lo, hi = max(j - 5, 0), min(j + 6, n)
        if not C[t - 12:t, lo:hi].any() and C[t, lo:hi].sum() == 1 and W <= j < n - W - 2:
            cands.append((t, j))
pick = [cands[i] for i in rng.choice(len(cands), min(K, len(cands)), replace=False)]
for t, j in pick:
    cols = range(j - W, j + W + 1)
    print(f"\n{panel} birth at link i{j} {P.links[j]} {P.times[t]}  (lanes/cap per col below; '#' = congested)")
    print("        " + " ".join(f"{c - j:+4d}" for c in cols))
    print("  lanes " + " ".join(f"{P.lanes[c]:4.0f}" for c in cols))
    print("  capk  " + " ".join(f"{P.cap[c] / 1000:4.1f}" for c in cols))
    print("  L(m)  " + " ".join(f"{P.length[c] * 1000:4.0f}" for c in cols))
    for tt in range(t - 3, t + 10):
        print(f"  {P.times[tt][11:16]} " + " ".join(("  . " if not np.isfinite(V[tt, c]) else (f"#{V[tt, c] / P.vf[c]:.2f}"[0:4] if C[tt, c] else f"{V[tt, c] / P.vf[c]:4.2f}")) for c in cols))
    print("  flow (k vph):")
    for tt in range(t - 3, t + 10, 3):
        print(f"  {P.times[tt][11:16]} " + " ".join(("  . " if not np.isfinite(Q[tt, c]) else f"{Q[tt, c] / 1000:4.1f}") for c in cols))
