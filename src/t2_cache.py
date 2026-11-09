"""Cache Task 2 per-window link features (priors excluded) for fast experiments.

Saves work/t2/featcache.npz: X (rows, f) float32 for every (window, link), window offsets,
plus per-window metadata. Prior columns are appended per fold at train time.
"""
from __future__ import annotations
import pickle, time
import numpy as np
from cachelib import load, T2_PANELS
from t2_model import PanelInputs, link_features
from t2_model2 import kin_features

import os
SUF = "_noT" if os.environ.get("TFB_T2_NOT", "0") == "1" else ""
OUT = rf"C:/Users/user/tfb/work/t2/featcache{SUF}.npz"


def window_X(PI, t, cond):
    zeros = np.zeros(PI.P.n)
    X1, n1 = link_features(PI, t, cond, zeros, zeros)
    X2, n2 = kin_features(PI, t)
    return np.column_stack([X1, X2]).astype(np.float32), n1 + n2


if __name__ == "__main__":
    W = pickle.load(open(r"C:/Users/user/tfb/work/t2_windows_train.pkl", "rb"))
    E = pickle.load(open(r"C:/Users/user/tfb/work/t2_windows_extra12.pkl", "rb")) + pickle.load(open(r"C:/Users/user/tfb/work/t2_windows_extra4.pkl", "rb"))
    ALL = [dict(w, is_eval=True) for w in W] + [dict(w, is_eval=False) for w in E]
    PIs = {p: PanelInputs(load(p, "train")) for p in T2_PANELS}
    t0 = time.time(); Xs = []; off = [0]; names = None
    for k, w in enumerate(ALL):
        X, names = window_X(PIs[w["panel"]], w["t"], w["cond"])
        Xs.append(X); off.append(off[-1] + len(X))
        if k % 5000 == 0:
            print(k, len(ALL), f"{time.time()-t0:.0f}s", flush=True)
    meta = dict(panel=np.array([w["panel"] for w in ALL]), t=np.array([w["t"] for w in ALL]), cond=np.array([w["cond"] for w in ALL]),
                date=np.array([w["date"] for w in ALL]), is_eval=np.array([w["is_eval"] for w in ALL]))
    np.savez(OUT, X=np.vstack(Xs), off=np.array(off), names=np.array(names), **meta)
    pickle.dump([(w["label"], w["elig"]) for w in ALL], open(rf"C:/Users/user/tfb/work/t2/labels_all{SUF}.pkl", "wb"))
    print("done", off[-1], "rows", f"{time.time()-t0:.0f}s")
