"""G27: value of the T row (allowed by the organizer ruling, discussion 742068: data with timestamp at or before T).
In the test months the T row comes from the Task 1-masked layer (48-67% observed). On train: the T row masked with
the released Task 1 blank mask of that row (P.blank) plus natural missingness. Official-like ongoing windows:
persistence from the last history row (T-5, unmasked) vs persistence from the T row where observed (else T-5).
Also onset windows: does the T row already show the T+30 queue? (fraction of onset windows with a queued T-row cell)"""
import sys, numpy as np, pandas as pd
sys.path.insert(0, r"C:/Users/user/tfb/src")
import t2_eval_ongoing_shift as O, t2_eval_surprise as S
from t2_eval_oof import T
from cachelib import load
per = np.load("E:/tfb_nn/persist_iou_afdem.npy"); PC = {}


def iou(s, y, el):
    u = ((s | y) & el).sum(); return 1.0 if u == 0 else (s & y & el).sum() / u


def P_(p):
    if p not in PC:
        P = load(p, "train"); PC[p] = (P, (np.nan_to_num(P.V, nan=1e9) <= P.vcut[None, :]) & P.E, np.isfinite(P.V) & ~P.blank & P.E)
    return PC[p]


R = []
for k in O.SEL:
    if per[k] > 0.9:
        continue
    p = str(T.META["panel"][k]); t = int(T.META["t"][k]); P, Q, OBS = P_(p)
    lab, el = T.LAB[k]; y = lab.astype(bool); el = el.astype(bool)
    last = Q[t - 1]; tr = np.where(OBS[t], Q[t], last); full = Q[t]
    F = y.shape[0]
    R.append(dict(p_tm5=iou(np.broadcast_to(last, y.shape), y, el), p_trow=iou(np.broadcast_to(tr, y.shape), y, el),
                  p_full=iou(np.broadcast_to(full, y.shape), y, el), obs_frac=OBS[t].mean()))
R = pd.DataFrame(R); print("official-like ongoing persistence IoU:", R[["p_tm5", "p_trow", "p_full"]].mean().round(4).to_dict(), "T-row observed", round(R.obs_frac.mean(), 3))
n = 0; hit = 0
for k in S.SEL:
    p = str(T.META["panel"][k]); t = int(T.META["t"][k]); P, Q, OBS = P_(p)
    n += 1; hit += bool((Q[t] & OBS[t]).any())
print(f"onset windows with a visible queued cell in the T row: {hit}/{n}")
