"""G194 (11/01, ledger #164): the row at T for the Task 2 networks. The columns v_T and q_T exist in X and are all NaN (NO_T). Here they are filled as the other history columns are (v / vf, q / cap): train = the Task 1 masked layer of PanelInputs (the day's mask on row T, as G27), March = the release's row T.
Writes C:/tfb_data/tfb_nn/trow/trow_train.npy (rows of X, 2) and the copy of the M3 inputs C:/tfb_data/tfb_nn/nn_test_m3v3t/. Identity: v_T1 of X against the same layer at row T - 1 on a sample of windows; the share of row T seen.
usage: py -B -X utf8 g194_trow_build.py"""
import os, sys, pickle, shutil, numpy as np, warnings
os.environ["TFB_T2_NOT"] = "1"; os.environ["TFB_T2_ALLFIN"] = "1"; warnings.filterwarnings("ignore"); sys.dont_write_bytecode = True; sys.path.insert(0, r"C:/Users/user/tfb/src")
from cachelib import load
from t2_model import PanelInputs
NN = "C:/tfb_data/tfb_nn/"; O = NN + "trow/"; os.makedirs(O, exist_ok=True); d = NN + "nn_data_afdem/"; X = np.load(d + "X.npy", mmap_mode="r"); names = [str(x) for x in np.load(d + "names.npy", allow_pickle=True)]; off = np.load(d + "off.npy"); pan = np.array([str(x) for x in np.load(d + "panel.npy", allow_pickle=True)]); t_ = np.load(d + "t.npy"); iv, iq, iv1, iq1 = [names.index(c) for c in ("v_T", "q_T", "v_T1", "q_T1")]
TR = np.full((X.shape[0], 2), np.nan, np.float32); seen = []; chk = []
for p in sorted(set(pan)):
    P = load(p, "train"); PI = PanelInputs(P); vr = PI.mask_v / P.vf[None, :]; qc = PI.mask_q / P.cap[None, :]; ks = np.flatnonzero(pan == p)
    for k in ks:
        a, b = int(off[k]), int(off[k + 1]); assert b - a == P.n, (k, b - a, P.n); TR[a:b, 0] = vr[int(t_[k])]; TR[a:b, 1] = qc[int(t_[k])]
    seen.append(float(np.isfinite(TR[off[ks[0]]:off[ks[0] + 1], 0]).mean()))
    for k in ks[:: max(len(ks) // 20, 1)]:
        a, b = int(off[k]), int(off[k + 1]); x = np.array(X[a:b, iv1]); r = vr[int(t_[k]) - 1]; f = np.isfinite(x) & np.isfinite(r); chk.append(float(np.abs(x[f] - r[f]).max()) if f.any() else 0.0)
    print(p, len(ks), flush=True); load.cache_clear()
assert np.isfinite(TR[:, 0]).any(); np.save(O + "trow_train.npy", TR); print("share of row T seen, train", round(float(np.isfinite(TR[:, 0]).mean()), 4), "| |v_T1 - layer at T - 1| max on the sample", round(max(chk), 6), flush=True)
m = pickle.load(open("C:/Users/user/tfb/work/t2/m3v3_meta.pkl", "rb"))["meta"]; M = {x["window_id"]: x for x in m}; d2 = NN + "nn_test_m3v3/"; D2 = NN + "nn_test_m3v3t/"; os.makedirs(D2, exist_ok=True)
for f in ("cond", "off", "window_id", "R"):
    if not os.path.exists(D2 + f + ".npy"):
        shutil.copy(d2 + f + ".npy", D2 + f + ".npy")
X2 = np.array(np.load(d2 + "X.npy", mmap_mode="r")); off2 = np.load(d2 + "off.npy"); wid = [str(w) for w in np.load(d2 + "window_id.npy", allow_pickle=True)]; C = {}
for i, w in enumerate(wid):
    a, b = int(off2[i]), int(off2[i + 1]); p = M[w]["panel"]; t = int(M[w]["t"])
    if p not in C:
        P = load(p, "validation"); PI = PanelInputs(P); C[p] = (PI.mask_v / P.vf[None, :], PI.mask_q / P.cap[None, :])
    assert b - a == C[p][0].shape[1]; X2[a:b, iv] = C[p][0][t]; X2[a:b, iq] = C[p][1][t]
np.save(D2 + "X.npy", X2); print("share of row T seen, M3", round(float(np.isfinite(X2[:, iv]).mean()), 4), "windows", len(wid), flush=True)
