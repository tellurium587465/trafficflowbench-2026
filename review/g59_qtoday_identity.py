"""Identity check of the TFB_T2_QTODAY_P75 switch in t2_model.link_features (10/22): the M3 build path (window_X, the
t2_m3_build.py settings) with the switch off must reproduce nn_test_m3 and with it on nn_test_m3q (g59_qtoday_m3.py),
every column, for a sample of windows.
usage: python g59_qtoday_identity.py <0|1>"""
import sys, os, pickle, numpy as np
FLAG = sys.argv[1]
os.environ["TFB_T2_NOT"] = "1"; os.environ["TFB_T2_ALLFIN"] = "1"; os.environ["TFB_T2_QTODAY_P75"] = FLAG
sys.path.insert(0, r"C:/Users/user/tfb/src")
from cachelib import load
from common import NN
from t2_model import PanelInputs
from t2_cache import window_X
from t2_arrival import arrival_features, _dmin
from t2_demand import demand_features, profile
names = [str(x) for x in np.load(f"{NN}/nn_data_afdem/names.npy", allow_pickle=True)]
I_PON, I_POG = names.index("prior_on"), names.index("prior_og")
S = f"{NN}/nn_test_m3" + ("q" if FLAG == "1" else "")
X = np.load(f"{S}/X.npy", mmap_mode="r"); off = np.load(f"{S}/off.npy"); wid = [str(w) for w in np.load(f"{S}/window_id.npy", allow_pickle=True)]
pos = {w: i for i, w in enumerate(wid)}
META = pickle.load(open("C:/Users/user/tfb/work/t2/m3_meta.pkl", "rb"))["meta"]
rng = np.random.default_rng(0); pick = [META[i] for i in rng.choice(len(META), 40, replace=False)]
PIs = {}; err = 0.0
for m in pick:
    if m["panel"] not in PIs:
        P = load(m["panel"], "validation"); PIs[m["panel"]] = (P, PanelInputs(P), profile(load(m["panel"], "train")))
    P, PI, prof = PIs[m["panel"]]; t = m["t"]                  # as t2_m3_build.py
    Xw, _ = window_X(PI, t, m["cond"]); Xw[:, I_PON] = 0.0; Xw[:, I_POG] = 0.0
    V, Q, tl = PI.window(t); Xa, _ = arrival_features(P, V, Q, tl, PI.x)
    Xd = demand_features(P, Q, tl, t, P.times[t][:10], prof, _dmin(P, PI.x))
    Xw = np.column_stack([Xw, Xa, Xd]).astype(np.float32)
    i = pos[m["window_id"]]; ref = np.asarray(X[off[i]:off[i + 1]])
    d = np.where(np.isnan(ref) & np.isnan(Xw), 0.0, np.abs(ref - Xw)); err = max(err, float(np.nanmax(d)))
print(f"switch {FLAG}: 40 windows vs {S}, max |diff| over all columns {err}")
