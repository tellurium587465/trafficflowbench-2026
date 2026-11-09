"""G204 (11/02, ledger #164): row T for the production networks: nn_test_afdemt (official windows, the release's row T) and nn_data_aprilt (April pool of X1); nn_data_marcht comes from G202. v_T, q_T are filled as the other history columns; everything else is copied. The row at T is at or before T, so the windows' reads stay causal.
usage: py -B -X utf8 g204_trow_official.py"""
import os, sys, shutil, numpy as np, pandas as pd, warnings
os.environ["TFB_T2_NOT"] = "1"; os.environ["TFB_T2_ALLFIN"] = "1"; warnings.filterwarnings("ignore"); sys.dont_write_bytecode = True; sys.path.insert(0, r"C:/Users/user/tfb/src")
from cachelib import load, T2_PANELS
from common import REL
from t2_model import PanelInputs
NN = "C:/tfb_data/tfb_nn/"; names = [str(x) for x in np.load(NN + "nn_data_afdem/names.npy", allow_pickle=True)]; iv, iq = names.index("v_T"), names.index("q_T"); C = {}


def layer(p, sp):
    if (p, sp) not in C:
        P = load(p, sp); PI = PanelInputs(P); C[(p, sp)] = (PI.mask_v / P.vf[None, :], PI.mask_q / P.cap[None, :], {str(t): i for i, t in enumerate(P.times)})
    return C[(p, sp)]


d = NN + "nn_data_april/"; O = NN + "nn_data_aprilt/"; os.makedirs(O, exist_ok=True)
for f in os.listdir(d):
    if f != "X.npy" and not os.path.exists(O + f):
        shutil.copy(d + f, O + f)
X = np.array(np.load(d + "X.npy", mmap_mode="r")); off = np.load(d + "off.npy"); pan = [str(x) for x in np.load(d + "panel.npy", allow_pickle=True)]; t_ = np.load(d + "t.npy"); assert np.isnan(X[:, iv]).all()
for k in range(len(off) - 1):
    a, b = int(off[k]), int(off[k + 1]); V, Q, _ = layer(pan[k], "private"); assert b - a == V.shape[1]; X[a:b, iv] = V[int(t_[k])]; X[a:b, iq] = Q[int(t_[k])]
np.save(O + "X.npy", X); print("April pool windows", len(off) - 1, "row T seen", round(float(np.isfinite(X[:, iv]).mean()), 4), flush=True)
d = NN + "nn_test_afdem/"; O = NN + "nn_test_afdemt/"; os.makedirs(O, exist_ok=True); print(sorted(os.listdir(d))[:8], flush=True)
for f in os.listdir(d):
    if f != "X.npy" and not os.path.exists(O + f):
        shutil.copy(d + f, O + f)
X = np.array(np.load(d + "X.npy", mmap_mode="r")); off = np.load(d + "off.npy"); wid = [str(w) for w in np.load(d + "window_id.npy", allow_pickle=True)]; R = {}
for p in T2_PANELS:
    for sp in ("validation", "private"):
        for r in pd.read_csv(REL / "task2" / p / sp / "window_index.csv").itertuples():
            R[r.window_id] = (p, sp, r.forecast_origin)
nf = []
for i, w in enumerate(wid):
    if w not in R:
        continue
    p, sp, fo = R[w]; V, Q, ti = layer(p, sp); t = ti[fo]; a, b = int(off[i]), int(off[i + 1]); assert b - a == V.shape[1], (w, b - a); X[a:b, iv] = V[t]; X[a:b, iq] = Q[t]; nf.append(float(np.isfinite(V[t]).mean()))
np.save(O + "X.npy", X); print("official windows", len(nf), "of", len(wid), "row T seen, mean", round(float(np.mean(nf)), 4), "range", round(min(nf), 3), round(max(nf), 3), flush=True)
