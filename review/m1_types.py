"""M1: label-defined memory-consistency types of the selector-like ongoing windows (train OOF), per type IoU / far rule.
A = a usual far region (U >= 0.3 at that slot) activates; B = usual far region(s) exist and all stay free;
C = only a rare far region (U < 0.05) activates; N = no usual far region and no far activation; M = other mixes.
Stationary = A + N (outcome agrees with the location-time memory); non-stationary = B + C (memory is wrong)."""
import sys; sys.argv = [sys.argv[0]]
import numpy as np, pandas as pd
import m1_msiw as S
spec = "l127_full_af:2+q14_attn1_glob+q16_attnglob_s1+q14_attn1_wp05:2+q13_attn1:2+q12_af_r13_nopos+q17_drop1"
R = S.per_window(spec); W = pd.read_pickle(r"C:/Users/user/tfb/review/m1_windows.pkl").set_index("k")
typ = []
for k in R.k:
    lab, el = S.LAB[k]; far = S.EV.far_mask(k); d = str(S.M["date"][k]); we = pd.Timestamp(d).dayofweek >= 5
    b = min((int(S.M["t"][k]) % 288 + 3) // 6, 47); Uk = S.U[(S.M["panel"][k], d[:7], we)][b]
    q = (lab.astype(bool) & el.astype(bool)).any(axis=0) & far
    hi = far & (Uk >= 0.3); lo = far & (Uk < 0.05)
    a = (q & hi).any(); r = (q & lo).any(); anyq = q.any()
    typ.append("A" if a else ("B" if hi.any() and not anyq else ("C" if r and not (q & ~lo).any() else ("N" if not hi.any() and not anyq else "M"))))
R["type"] = typ; R["d"] = R.far - R.iou
R = R.join(W[["far_pred", "far_tp", "n_pred"]], on="k")
g = R.groupby("type").agg(n=("iou", "size"), iou=("iou", "mean"), far=("far", "mean"), d=("d", "mean"), fp=("far_pred", "sum"), tp=("far_tp", "sum"))
g["share"] = g.n / g.n.sum(); g["prec"] = g.tp / g.fp
print(g.round(4).to_string())
R[["k", "type"]].to_pickle(r"C:/Users/user/tfb/review/m1_types.pkl")
