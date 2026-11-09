"""Standalone value of the P1 physics features on selector-like train windows (AUC vs labels), timing included."""
import os, time, warnings, numpy as np
os.environ.setdefault("TFB_T2_NOT", "1"); os.environ.setdefault("TFB_T2_ALLFIN", "1"); warnings.filterwarnings("ignore")
from cachelib import load
from t2_model import PanelInputs
from t2_fdmargin import fdmargin_features, NAMES
from t2_eval_oof import T


def auc(s, y):
    o = np.argsort(s, kind="stable"); r = np.empty(len(s)); r[o] = np.arange(len(s)); n1 = y.sum(); n0 = len(y) - n1
    return (r[y].sum() - n1 * (n1 - 1) / 2) / max(n1 * n0, 1)


PIs = {}; t0 = time.time()
for cond, step in (("queue_ongoing", 0), ("queue_ongoing", -1)):
    ks = [k for k in range(T.NWIN) if T.META["is_sel"][k] and T.META["cond"][k] == cond][::2]
    S, Y = [], []
    for k in ks:
        p = str(T.META["panel"][k])
        if p not in PIs:
            PIs[p] = PanelInputs(load(p, "train"))
        PI = PIs[p]; V, Q, tl = PI.window(int(T.META["t"][k])); X = fdmargin_features(PI.P, V, Q, tl)
        lab, el = T.LAB[k]; e = el[step].astype(bool); S.append(X[e]); Y.append(lab[step].astype(bool)[e])
    S = np.vstack(S); Y = np.concatenate(Y)
    inb = np.isfinite(S[:, 2]); S = S[inb]; Y = Y[inb]
    print(f"{cond} step {step}: within-band cells {inb.sum()}, positives {Y.mean():.3f}, {time.time()-t0:.0f}s")
    for j, n in enumerate(NAMES):
        s = np.nan_to_num(S[:, j], nan=np.nanmedian(S[:, j]))
        a = auc(s, Y); print(f"   AUC {n:14s} {max(a, 1 - a):.4f} ({'low=queue' if a < 0.5 else 'high=queue'})  finite {np.isfinite(S[:, j]).mean():.2f}")
