"""Export validation/private Task 2 windows in the NN input format (same columns as nn_data.npz)."""
import os, numpy as np, pandas as pd
os.environ.setdefault("TFB_T2CACHE", "2_noT"); os.environ.setdefault("TFB_T2_ARR", "1"); os.environ.setdefault("TFB_T2_NOT", "1")
import t2_fast as T
from cachelib import load, T2_PANELS
from common import REL
from t2_model import PanelInputs
from t2_cache import window_X
from t2_arrival import arrival_features

pr = T.priors(set())
Xs, off, wid, cond = [], [0], [], []
for split in ("validation", "private"):
    for panel in T2_PANELS:
        P = load(panel, split); PI = PanelInputs(P); ti = {s: i for i, s in enumerate(P.times)}
        for r in pd.read_csv(REL / "task2" / panel / split / "window_index.csv").itertuples():
            t = ti[r.forecast_origin]
            X, _ = window_X(PI, t, r.condition); X[:, T.I_PON] = pr[panel][0]; X[:, T.I_POG] = pr[panel][1]
            V, Q, tl = PI.window(t); Xa, _ = arrival_features(P, V, Q, tl, PI.x)
            Xs.append(np.column_stack([X, Xa]).astype(np.float32)); off.append(off[-1] + len(X))
            wid.append(r.window_id); cond.append(r.condition)
np.savez(r"C:/Users/user/tfb/work/t2/nn_test.npz", X=np.vstack(Xs), off=np.array(off), window_id=np.array(wid), cond=np.array(cond))
print("exported", off[-1], len(wid))
