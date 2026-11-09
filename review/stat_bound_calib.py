"""Calibrate the Jaccard-distance bound on OOF (selector-like windows): how big is |dIoU| relative to d(A,B)?"""
import sys, os, numpy as np
sys.path.insert(0, r"C:/Users/user/tfb/src"); os.chdir(r"C:/Users/user/tfb/src")
from t2_eval_oof import T, load_oof
from t2_model import best_set
sel = [k for k in range(T.NWIN) if T.META["is_sel"][k]]
def sets(spec):
    o = load_oof(spec); return {k: best_set(o[k].ravel(), np.ones(o[k].size, bool)).reshape(o[k].shape) for k in sel if k in o}
def iou(s, lab, el):
    u = ((s | lab) & el).sum(); return 1.0 if u == 0 else (s & lab & el).sum() / u
S = {n: sets(s) for n, s in (("LGB", "l127_noT_arr"), ("BLEND", "l127_noT_arr+q12_af_r13_nopos"), ("NN", "q12_af_r13_nopos"),
                             ("NNq10", "q10_raw"), ("NNq10np", "q10_raw_nopos"))}
for a, c in (("LGB", "BLEND"), ("NN", "BLEND"), ("NNq10", "NNq10np"), ("LGB", "NN")):
    for cd in ("queue_onset", "queue_ongoing"):
        ks = [k for k in sel if T.META["cond"][k] == cd and k in S[a] and k in S[c]]
        d = np.array([1 - (S[a][k] & S[c][k]).sum() / max((S[a][k] | S[c][k]).sum(), 1) for k in ks])
        dl = np.array([iou(S[c][k], *T.LAB[k]) - iou(S[a][k], *T.LAB[k]) for k in ks])
        big = d > 0.5
        print(f"{a:5s}->{c:7s} {cd[6:]:7s} mean d {d.mean():.3f}  rms(dIoU)/rms(d) {np.sqrt((dl**2).mean()/(d**2).mean()):.2f}  "
              f"mean dIoU {dl.mean():+.4f}  on d>0.5 (n={big.sum()}): mean|dIoU| {np.abs(dl[big]).mean() if big.any() else np.nan:.2f}, "
              f"B wins {np.mean(dl[big]>0) if big.any() else np.nan:.2f}")
