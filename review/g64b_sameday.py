"""G64b: is the rare onset site visible earlier the same day? For the dev onset windows of G64 (g64_onset_rare_diag.csv),
q_today (queued rows on the link earlier today, before T-12; nn_test_m3 X column) on the true links vs the other links,
+-2 links, by rare / common truth and for the lost windows; also 'today had any queue on the corridor'.
usage: python g64b_sameday.py"""
import sys, pickle, numpy as np, pandas as pd
sys.path.insert(0, r"C:/Users/user/tfb/src")
from common import NN
W = "C:/Users/user/tfb/work/t2/"
D = pd.read_csv(W + "g64_onset_rare_diag.csv")
META = {m["window_id"]: m for m in pickle.load(open(W + "m3_meta.pkl", "rb"))["meta"]}
names = [str(x) for x in np.load(f"{NN}/nn_data_afdem/names.npy", allow_pickle=True)]; IQ = names.index("q_today")
X = np.load(f"{NN}/nn_test_m3/X.npy", mmap_mode="r"); off = np.load(f"{NN}/nn_test_m3/off.npy")
pos = {str(w): i for i, w in enumerate(np.load(f"{NN}/nn_test_m3/window_id.npy", allow_pickle=True))}
rows = []
for r in D.itertuples():
    m = META[r.w]; i = pos[r.w]; qt = np.asarray(X[off[i]:off[i + 1], IQ]); n = len(qt)
    y = m["y"][:, 5].astype(bool) & m["m"][:, 5].astype(bool); tl = np.flatnonzero(y)
    near = np.zeros(n, bool)
    for j in tl:
        near[max(0, j - 2):j + 3] = True
    rows.append(dict(w=r.w, rare=r.rare_true, lost=r.iou < 0.2, true_seen=bool((qt[tl] > 0).any()), near_seen=bool((qt[near] > 0).any()),
                     other_rate=float((qt[~near] > 0).mean()) if (~near).any() else np.nan, any_today=bool((qt > 0).any()),
                     slot=(m["t"] % 288) // 12))
R = pd.DataFrame(rows)
for name, g in (("rare truth", R[R.rare]), ("common truth", R[~R.rare]), ("lost", R[R.lost])):
    print(f"{name}: n {len(g)}, true link queued earlier today {g.true_seen.mean():.2f}, within +-2 {g.near_seen.mean():.2f}, "
          f"base rate on other links {g.other_rate.mean():.3f}, any queue on the corridor today {g.any_today.mean():.2f}, hour median {g.slot.median():.0f}")
print(R[R.lost].to_string(index=False))
