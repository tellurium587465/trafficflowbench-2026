"""Audit A9: the horizon cells' eligibility flags are released in val/private (quality fields stay inside the gap).
How much does restricting the expected-IoU set selection to eligible cells change the score?
(OOF of the LightGBM run l127_noT_arr_dem, selector-like and random eval windows.)"""
import sys, pickle, numpy as np
sys.path.insert(0, r"C:/Users/user/tfb/src")
from t2_windows import score
from t2_model import best_set

W = "C:/Users/user/tfb/work/t2/"
Z = np.load(W + "featcache2_noT.npz", allow_pickle=True)
meta = {k: Z[k] for k in ("panel", "cond", "is_eval", "is_sel")}
LAB = pickle.load(open(W + "labels_all2_noT.pkl", "rb"))
for tag in sys.argv[1:] or ["l127_noT_arr_dem"]:
    oof = pickle.load(open(W + f"oof_{tag}.pkl", "rb"))
    for key in ("is_eval", "is_sel"):
        idx = [k for k in oof if meta[key][k]]
        Wv = [dict(panel=meta["panel"][k], cond=meta["cond"][k], label=LAB[k][0], elig=LAB[k][1]) for k in idx]
        a = [best_set(oof[k].ravel(), np.ones(oof[k].size, bool)).reshape(oof[k].shape) for k in idx]
        b = [best_set(oof[k].ravel(), LAB[k][1].ravel()).reshape(oof[k].shape) for k in idx]
        sa, sb = score(Wv, a), score(Wv, b)
        print(f"{tag} {key}: n={len(idx)}  all-cells {np.round(sa,4)}  eligible-only {np.round(sb,4)}  delta {np.round(np.array(sb)-np.array(sa),4)}")
