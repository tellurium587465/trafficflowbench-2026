"""NN inputs with every finite released value (audit F4): train from the af caches (featcache2_noT_af + arrcache
+ demcache, same rows as nn_data), validation/private recomputed with TFB_T2_ALLFIN=1 in nn_test window order.
-> E:/tfb_nn/nn_data_afdem, E:/tfb_nn/nn_test_afdem (80 columns, same layout as nn_data_dem)."""
import os, shutil, warnings, numpy as np, pandas as pd
os.environ["TFB_T2_NOT"] = "1"; os.environ["TFB_T2_ALLFIN"] = "1"; warnings.filterwarnings("ignore")
from cachelib import load, T2_PANELS
from common import REL
from t2_model import PanelInputs
from t2_cache import window_X
from t2_arrival import arrival_features, _dmin
from t2_demand import demand_features, profile

W = "C:/Users/user/tfb/work/t2/"
# --- train: same rows as nn_data (featcache2_noT order) ---
Z = np.load(W + "featcache2_noT_af.npz", allow_pickle=True)
X = Z["X"]; A = np.load(W + "arrcache2_noT_af.npy"); Dm = np.load(W + "demcache2_noT_af.npy")
src, dst = "E:/tfb_nn/nn_data_dem", "E:/tfb_nn/nn_data_afdem"; os.makedirs(dst, exist_ok=True)
assert len(X) == len(np.load(f"{src}/X.npy", mmap_mode="r"))
out = np.lib.format.open_memmap(f"{dst}/X.npy", mode="w+", dtype=np.float32, shape=(len(X), 80))
for a in range(0, len(X), 1 << 21):
    out[a:a + (1 << 21)] = np.column_stack([X[a:a + (1 << 21)], A[a:a + (1 << 21)], Dm[a:a + (1 << 21)]])
out.flush()
for f in os.listdir(src):
    if f != "X.npy":
        shutil.copyfile(f"{src}/{f}", f"{dst}/{f}")
print("train ok", out.shape, flush=True)
# --- validation/private: recompute with all finite inputs, priors as in t2_nn_export_test (zeroed by the NN) ---
names = list(Z["names"]); I_PON, I_POG = names.index("prior_on"), names.index("prior_og")   # zeroed by the NN anyway
rows, ids = [], []
for split in ("validation", "private"):
    for p in T2_PANELS:
        P = load(p, split); PI = PanelInputs(P); ti = {s: i for i, s in enumerate(P.times)}; prof = profile(load(p, "train"))
        for r in pd.read_csv(REL / "task2" / p / split / "window_index.csv").itertuples():
            t = ti[r.forecast_origin]
            Xw, _ = window_X(PI, t, r.condition); Xw[:, I_PON] = 0.0; Xw[:, I_POG] = 0.0
            V, Q, tl = PI.window(t); Xa, _ = arrival_features(P, V, Q, tl, PI.x)
            Xd = demand_features(P, Q, tl, t, P.times[t][:10], prof, _dmin(P, PI.x))
            rows.append(np.column_stack([Xw, Xa, Xd]).astype(np.float32)); ids.append(r.window_id)
tsrc, tdst = "E:/tfb_nn/nn_test_dem", "E:/tfb_nn/nn_test_afdem"; os.makedirs(tdst, exist_ok=True)
assert ids == [str(w) for w in np.load(f"{tsrc}/window_id.npy", allow_pickle=True)]
Xt = np.vstack(rows); assert Xt.shape == np.load(f"{tsrc}/X.npy").shape, Xt.shape
np.save(f"{tdst}/X.npy", Xt)
for f in os.listdir(tsrc):
    if f != "X.npy":
        shutil.copyfile(f"{tsrc}/{f}", f"{tdst}/{f}")
print("test ok", Xt.shape)
