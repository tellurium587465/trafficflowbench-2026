"""q_today train/test filter (review B, strategy §(b) 1; 10/22): training windows count today's queued rows on Vm =
finite & pct >= 75 (t1_features.inputs), the test side (TFB_T2_ALLFIN=1) on every finite row -> the test counts more
observed rows. Inference-only check: recompute q_today for the 407 M3 windows with the training filter, write
NN/nn_test_m3q (X with only that column replaced), then re-infer saved models and compare on M3 v2.
Identity check first: the ALLFIN recount must equal the stored column.
usage: python g59_qtoday_m3.py            (writes NN/nn_test_m3q)"""
import sys, os, shutil, pickle, numpy as np
sys.path.insert(0, r"C:/Users/user/tfb/src")
from cachelib import load
from common import NN
from t2_windows import H

META = pickle.load(open("C:/Users/user/tfb/work/t2/m3_meta.pkl", "rb"))["meta"]
S = f"{NN}/nn_test_m3"; D = f"{NN}/nn_test_m3q"
names = [str(x) for x in np.load(f"{NN}/nn_data_afdem/names.npy", allow_pickle=True)]; IQ = names.index("q_today")
X = np.load(f"{S}/X.npy"); off = np.load(f"{S}/off.npy"); wid = [str(w) for w in np.load(f"{S}/window_id.npy", allow_pickle=True)]
pos = {w: i for i, w in enumerate(wid)}
cache = {}
old_err, ratio, n_old, n_new = 0.0, [], 0.0, 0.0
for m in META:
    p = m["panel"]
    if p not in cache:
        P = load(p, "validation"); fin = np.isfinite(P.V) & np.isfinite(P.Q)
        cache[p] = (P, fin, fin & (P.P >= 75), P.V <= P.vcut[None, :])
    P, fin, good, q = cache[p]
    t = m["t"]; d0 = t - (t % 288); hi = t - H          # rows d0 .. t-H-1, as link_features (vr[:tl - H])
    old = (fin[d0:hi] & q[d0:hi]).sum(axis=0).astype(np.float32) if hi > d0 else np.zeros(P.n, np.float32)
    new = (good[d0:hi] & q[d0:hi]).sum(axis=0).astype(np.float32) if hi > d0 else np.zeros(P.n, np.float32)
    i = pos[m["window_id"]]; a, b = off[i], off[i + 1]
    old_err = max(old_err, float(np.abs(X[a:b, IQ] - old).max()))
    n_old += old.sum(); n_new += new.sum()
    X[a:b, IQ] = new
print(f"identity (ALLFIN recount vs stored column): max |diff| {old_err}")
print(f"queued-row count today: test filter {n_old:.0f} -> training filter {n_new:.0f} (x{n_new / max(n_old, 1):.3f})")
os.makedirs(D, exist_ok=True)
for f in os.listdir(S):
    if f != "X.npy":
        shutil.copy(f"{S}/{f}", f"{D}/{f}")
np.save(f"{D}/X.npy", X); print("wrote", D)
