"""Standalone value of the CTM rollout: AUC / rule IoU on train windows vs the arrival-flow feature."""
import os, pickle, time, numpy as np
from cachelib import load
from t2_model import PanelInputs
from t2_ctm import ctm_features

C = os.environ.get("TFB_T2CACHE", "2_noT")
Z = np.load(rf"C:/Users/user/tfb/work/t2/featcache{C}.npz", allow_pickle=True)
off, panel, tt, cond = Z["off"], Z["panel"], Z["t"], Z["cond"]
sel = Z["is_sel"] if "is_sel" in Z.files else Z["is_eval"]
LAB = pickle.load(open(rf"C:/Users/user/tfb/work/t2/labels_all{C}.pkl", "rb"))
ARR = np.load(rf"C:/Users/user/tfb/work/t2/arrcache{C}.npy", mmap_mode="r")
I_ARR = 6                                            # arr30_over_dnmin


def auc(s, y):
    o = np.argsort(s); r = np.empty(len(s)); r[o] = np.arange(len(s)); n1 = y.sum(); n0 = len(y) - n1
    return (r[y].sum() - n1 * (n1 - 1) / 2) / max(n1 * n0, 1)


rng = np.random.default_rng(0); PIs = {}; t0 = time.time()
for cn in ("queue_onset", "queue_ongoing"):
    ks = np.flatnonzero(sel & (cond == cn)); ks = rng.choice(ks, min(1500, len(ks)), replace=False)
    S_ctm, S_arr, Y, iou = [], [], [], []
    for k in ks:
        p = str(panel[k]); PI = PIs.setdefault(p, PanelInputs(load(p, "train")))
        V, Q, tl = PI.window(int(tt[k])); X, _ = ctm_features(PI.P, V, Q, tl)
        lab, el = LAB[k]; y = lab[-1].astype(bool) & el[-1].astype(bool); e = el[-1].astype(bool)
        S_ctm.append(-X[e, 2]); S_arr.append(ARR[off[k]:off[k + 1], I_ARR][e]); Y.append(y[e])
        pred = X[:, 2] <= (PI.P.vcut / PI.P.vf); a = (pred & e & y).sum(); b = ((pred & e) | y).sum()
        iou.append(1.0 if b == 0 else a / b)
    Y = np.concatenate(Y); print(cn, len(ks), "windows", f"pos {Y.mean():.4f}",
                                 f"AUC ctm_vr30 {auc(np.concatenate(S_ctm), Y):.4f}",
                                 f"AUC arr30/dnmin {auc(np.nan_to_num(np.concatenate(S_arr)), Y):.4f}",
                                 f"rule IoU@T+30 {np.mean(iou):.4f}", f"{time.time()-t0:.0f}s", flush=True)
