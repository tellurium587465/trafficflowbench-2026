"""TE-A3: raw time series around a bottleneck during one queue episode + speed distribution of one link.
Usage: python te_a3_episode.py PANEL LINK_IDX lo hi [nth_episode]"""
import sys, numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, r"C:/Users/user/tfb/src")
from cachelib import load

panel, h, lo, hi = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
nth = int(sys.argv[5]) if len(sys.argv) > 5 else 0
P = load(panel, "train")
V, Q = P.V, P.Q
vr = V[:, h] / P.vf[h]
print("link", h, P.links[h], "v/vf quantiles all:", np.round(np.nanpercentile(vr, [0.1, 1, 5, 25, 50, 75, 95, 99, 99.9]), 4))
qr = Q[:, h] / P.cap[h]
for a, b in [(0, .2), (.2, .4), (.4, .5), (.5, .6), (.6, .7), (.7, 1.2)]:
    m = (qr >= a) & (qr < b)
    if m.sum():
        print(f"  q/cap in [{a},{b}): n={m.sum()} v/vf p5/50/95 {np.round(np.nanpercentile(vr[m], [5, 50, 95]), 4)}")
C = np.isfinite(V) & (V <= P.vcut[None, :])
on = np.flatnonzero(C[1:, h - 1] & ~C[:-1, h - 1]) + 1
t0 = on[min(nth, len(on) - 1)]
print("episode start", P.times[t0])
pd.set_option("display.width", 250)
cols = list(range(lo, hi + 1))
tab = []
for t in range(t0 - 4, t0 + 16):
    tab.append([P.times[t][11:16]] + [f"{V[t, i] / P.vf[i]:.2f}/{Q[t, i] / P.cap[i]:.2f}" for i in cols])
print(pd.DataFrame(tab, columns=["t"] + [f"i{i}" for i in cols]).to_string(index=False))
print("raw flow vph:")
tab = []
for t in range(t0 - 4, t0 + 16):
    tab.append([P.times[t][11:16]] + [f"{Q[t, i]:.0f}" for i in cols])
print(pd.DataFrame(tab, columns=["t"] + [f"i{i}" for i in cols]).to_string(index=False))
