import pickle, numpy as np
from collections import defaultdict
W2 = r"C:/Users/user/tfb/work/t2/"
Z = np.load(W2 + "featcache2_noT_af.npz", allow_pickle=True)
M = {k: Z[k] for k in ("panel", "cond", "is_sel")}
LAB = pickle.load(open(W2 + "labels_all2_noT_af.pkl", "rb"))
tr = np.flatnonzero(~M["is_sel"])
qy = np.array([(LAB[k][0] & LAB[k][1]).sum() for k in tr]); ne = np.array([LAB[k][1].sum() for k in tr])
ne1 = np.array([LAB[k][1][-1].sum() for k in tr])
cond = M["cond"][tr]; pan = M["panel"][tr]
on = cond == "queue_onset"
print("median |Y| onset/ongoing:", np.median(qy[on]), np.median(qy[~on]))
for wp in (0.0, 0.5, 1.0):
    wb = 100.0 * np.maximum(qy, 3) ** (-wp)
    lw = wb * ne                      # NN: BCE summed over all eligible cells of the 6 steps
    print(f"WPOW={wp}: onset share of NN loss weight {lw[on].sum()/lw.sum():.3f} | per-window weight ratio onset/ongoing {wb[on].mean()/wb[~on].mean():.1f}")
    ps = defaultdict(float)
    for p, x in zip(pan, lw): ps[p] += x
    tot = sum(ps.values()); print("   panel shares:", {p[:9]: round(v/tot, 3) for p, v in sorted(ps.items())})
