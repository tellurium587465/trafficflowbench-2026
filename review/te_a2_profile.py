"""TE-A2: spatial profile around a bottleneck while a queue is active (train, unmasked).
Usage: python te_a2_profile.py PANEL HEAD_IDX [lo hi]"""
import sys, numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, r"C:/Users/user/tfb/src")
from cachelib import load

panel, h = sys.argv[1], int(sys.argv[2])
lo = int(sys.argv[3]) if len(sys.argv) > 3 else h - 4
hi = int(sys.argv[4]) if len(sys.argv) > 4 else h + 4
P = load(panel, "train")
V, Q = P.V, P.Q
C = np.isfinite(V) & (V <= P.vcut[None, :])
# times where h congested and h+1 not congested and h+1 finite
sel = C[:, h] & np.isfinite(V[:, h + 1]) & ~C[:, h + 1]
idx = np.flatnonzero(sel)
print(panel, "head", h, "times", len(idx))
pd.set_option("display.width", 250)
rows = []
for i in range(lo, hi + 1):
    v = V[idx, i] / P.vf[i]; q = Q[idx, i]
    k = q / np.maximum(V[idx, i], 1)
    rows.append(dict(i=i, link=P.links[i], L=P.length[i], lanes=P.lanes[i], vf=P.vf[i], cap=P.cap[i], capL=P.cap[i] / P.lanes[i],
                     kc=P.kc[i], kj=P.kj[i], w=P.w[i], on=P.on_attached[i], off=P.off_attached[i],
                     v_vf_med=np.nanmedian(v), q_med=np.nanmedian(q), q_cap=np.nanmedian(q) / P.cap[i], k_kc=np.nanmedian(k) / P.kc[i],
                     cong_share=np.nanmean(C[idx, i])))
print(pd.DataFrame(rows).round(3).to_string(index=False))
# FD consistency at h+1: is (k, q) on the triangular FD?
i = h + 1
k = Q[idx, i] / np.maximum(V[idx, i], 1)
qfd = np.where(k <= P.kc[i], P.vf[i] * k, P.w[i] * (P.kj[i] - k))
print("h+1: median q/qFD", np.nanmedian(Q[idx, i] / qfd), " k/kc p10/50/90", np.nanpercentile(k / P.kc[i], [10, 50, 90]))
# also distribution of v/vf at h+1 over all times when q/cap>0.6
allv = V[:, i] / P.vf[i]; allq = Q[:, i] / P.cap[i]
m = np.isfinite(allv) & (allq > 0.6)
print("h+1 all times q/cap>0.6: v/vf p10/50/90", np.nanpercentile(allv[m], [10, 50, 90]), "n", m.sum())
