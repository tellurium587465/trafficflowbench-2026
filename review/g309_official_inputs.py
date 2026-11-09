"""G309 (11/05, ledger #246 step 1, development copy of the new-month stage): the official Task 2 windows' network inputs rebuilt from the
release (window_index.csv + the split's files), the per-window features as G220 / G296 (window_X for the window's condition, the priors
zeroed, arrival + demand columns; TFB_T2_NOT=1, TFB_T2_ALLFIN=1, history as released), written in the test layout (X, off, window_id, cond)
to NN/<out>. Identity: against the shipped nn_test_afdem on the same windows. usage: py -B -X utf8 g309_official_inputs.py <out> <split> [panel ...]"""
import os, sys, numpy as np, pandas as pd, warnings
os.environ["TFB_T2_NOT"] = "1"; os.environ["TFB_T2_ALLFIN"] = "1"; warnings.filterwarnings("ignore")
sys.path.insert(0, r"C:/Users/user/tfb/src")
from cachelib import load, T2_PANELS
from common import REL, NN
from t2_model import PanelInputs
from t2_cache import window_X
from t2_arrival import arrival_features, _dmin
from t2_demand import demand_features, profile
out, split = sys.argv[1], sys.argv[2]; panels = sys.argv[3:] or list(T2_PANELS)
names = [str(x) for x in np.load(f"{NN}/nn_data_afdem/names.npy", allow_pickle=True)]; XS, OF, WID, CND = [], [0], [], []
for p in panels:
    P = load(p, split); PI = PanelInputs(P); prof = profile(load(p, "train")); ti = {s: i for i, s in enumerate(P.times)}
    for r in pd.read_csv(REL / "task2" / p / split / "window_index.csv").itertuples():
        t = ti[r.forecast_origin]; Xw, _ = window_X(PI, t, r.condition); Xw[:, names.index("prior_on")] = 0.0; Xw[:, names.index("prior_og")] = 0.0
        V, Q, tl = PI.window(t); Xa, _ = arrival_features(P, V, Q, tl, PI.x); Xd = demand_features(P, Q, tl, t, P.times[t][:10], prof, _dmin(P, PI.x))
        X = np.column_stack([Xw, Xa, Xd]).astype(np.float32); XS.append(X); OF.append(OF[-1] + len(X)); WID.append(r.window_id); CND.append(r.condition)
    load.cache_clear()
o = f"{NN}/{out}/"; os.makedirs(o, exist_ok=True); np.save(o + "X.npy", np.vstack(XS)); np.save(o + "off.npy", np.array(OF)); np.save(o + "window_id.npy", np.array(WID)); np.save(o + "cond.npy", np.array(CND))
print("built", out, split, len(WID), "windows", OF[-1], "rows", XS[0].shape[1], "columns", flush=True)
ref = f"{NN}/nn_test_afdem/"
if os.path.exists(ref + "X.npy"):
    RX = np.load(ref + "X.npy", mmap_mode="r"); RO = np.load(ref + "off.npy"); RW = {str(w): i for i, w in enumerate(np.load(ref + "window_id.npy", allow_pickle=True))}; Xn = np.vstack(XS); mx = np.zeros(Xn.shape[1]); n = 0
    for i, w in enumerate(WID):
        if w in RW:
            j = RW[w]; a = np.nan_to_num(Xn[OF[i]:OF[i + 1]], nan=-9.0); b = np.nan_to_num(np.asarray(RX[RO[j]:RO[j + 1]], np.float32), nan=-9.0); assert a.shape == b.shape, (w, a.shape, b.shape); mx = np.maximum(mx, np.abs(a - b).max(axis=0)); n += 1
    top = np.argsort(-mx)[:5]; print(f"identity against nn_test_afdem on {n} windows: max |d| {mx.max():.2e}; largest columns {[(names[k], float(mx[k])) for k in top]}", flush=True)
