"""M1: ongoing level and the far-rule delta under different CV schemes (month / panel LOPO / spatial block), selector-like
ongoing windows (is_sel). Anchor 2 says the far rule must not hurt (LB +0.021); anchor 1 says the level is ~0.82.
usage: python m1_cvcheck.py spec1 spec2 ...
"""
import os, sys
os.environ["TFB_T2CACHE"] = "2_noT_af"; os.environ["TFB_T2_FAR"] = "3"
sys.path.insert(0, r"C:/Users/user/tfb/src")
import numpy as np, pandas as pd
import t2_eval_oof as EV
from t2_model import best_set
from t2_windows import iou

M, LAB = EV.T.META, EV.T.LAB
OG = [k for k in range(EV.T.NWIN) if M["is_sel"][k] and M["cond"][k] == "queue_ongoing"]
ON = [k for k in range(EV.T.NWIN) if M["is_sel"][k] and M["cond"][k] == "queue_onset"]


def run(spec):
    o = EV.load_oof(spec); r = []
    for k in OG:
        if k not in o:
            continue
        s = best_set(o[k].ravel(), np.ones(o[k].size, bool)).reshape(o[k].shape); far = EV.far_mask(k)
        lab, el = LAB[k]; lab = lab.astype(bool); el = el.astype(bool)
        r.append((M["panel"][k], iou(s, lab, el), iou(s & ~far[None, :], lab, el)))
    d = pd.DataFrame(r, columns=["panel", "iou", "far"]); g = d.groupby("panel").mean()
    on = []
    for k in ON:
        if k in o:
            s = best_set(o[k].ravel(), np.ones(o[k].size, bool)).reshape(o[k].shape); lab, el = LAB[k]
            on.append((M["panel"][k], iou(s, lab.astype(bool), el.astype(bool))))
    on = pd.DataFrame(on, columns=["panel", "iou"]).groupby("panel").iou.mean().mean() if on else np.nan
    return len(d), g.iou.mean(), g.far.mean(), on


if __name__ == "__main__":
    for s in sys.argv[1:]:
        n, a, b, on = run(s)
        print(f"{s[:60]:60s} n={n:4d} ongoing {a:.4f} far-rule {b:.4f} (d {b - a:+.4f})  onset {on:.4f}", flush=True)
