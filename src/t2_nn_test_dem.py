"""Append demand-profile columns to the validation/private NN inputs (same window order as t2_nn_export_test.py)
-> E:/tfb_nn/nn_test_dem. Profiles from all train months; only rows <= T-5 of the window's own day are used."""
import os, shutil, numpy as np, pandas as pd
os.environ.setdefault("TFB_T2_NOT", "1")
from cachelib import load, T2_PANELS
from common import REL
from t2_model import PanelInputs
from t2_arrival import _dmin
from t2_demand import demand_features, profile, NAMES

src, dst = "E:/tfb_nn/nn_test", "E:/tfb_nn/nn_test_dem"
os.makedirs(dst, exist_ok=True)
X = np.load(f"{src}/X.npy"); off = np.load(f"{src}/off.npy"); wid = np.load(f"{src}/window_id.npy", allow_pickle=True)
rows, ids = [], []
for split in ("validation", "private"):
    for p in T2_PANELS:
        P = load(p, split); PI = PanelInputs(P); ti = {s: i for i, s in enumerate(P.times)}; pr = profile(load(p, "train"))
        for r in pd.read_csv(REL / "task2" / p / split / "window_index.csv").itertuples():
            t = ti[r.forecast_origin]; V, Q, tl = PI.window(t)
            rows.append(demand_features(P, Q, tl, t, P.times[t][:10], pr, _dmin(P, PI.x))); ids.append(r.window_id)
assert list(ids) == [str(w) for w in wid], "window order differs from nn_test"
DEM = np.vstack(rows); assert len(DEM) == len(X)
np.save(f"{dst}/X.npy", np.column_stack([X, DEM]).astype(np.float32))
for f in os.listdir(src):
    if f != "X.npy":
        shutil.copyfile(f"{src}/{f}", f"{dst}/{f}")
print("ok", X.shape, "->", (len(X), X.shape[1] + DEM.shape[1]), NAMES)
