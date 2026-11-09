"""Review: ongoing IoU by whether the queue sits at a recurrent site, and where ongoing errors are (growth vs shrink)."""
import os, sys, numpy as np
from collections import defaultdict
sys.path.insert(0, r"C:/Users/user/tfb/src")
os.environ.setdefault("TFB_T2CACHE", "2_noT")
import t2_eval_oof as E
from t2_model import best_set, FOLDS

M = E.T.META; LAB = E.T.LAB; cond = M["cond"]; panel = M["panel"]
oL = E.oof_of("l127_fast_af"); oN = E.oof_of("q12_af_r13_nopos")
keys = sorted(set(oL) & set(oN)); PB = {k: 0.5 * oL[k] + 0.5 * oN[k] for k in keys}
# recurrent-site mask per panel: links queued in >= 5% of all ongoing labels (all train windows; coarse)
fr = {}
for k in range(E.T.NWIN):
    if cond[k] != "queue_ongoing": continue
    a = fr.setdefault(panel[k], [0, 0]); a[0] = a[0] + LAB[k][0].any(axis=0); a[1] += 1
rec = {p: np.convolve(s / n, np.ones(5), "same") >= 0.10 for p, (s, n) in fr.items()}
iou = lambda s, lab, el: 1.0 if ((s | lab) & el).sum() == 0 else ((s & lab & el).sum()) / ((s | lab) & el).sum()
res = defaultdict(list); trend = defaultdict(list)
for k in keys:
    if cond[k] != "queue_ongoing" or not M["is_sel"][k]: continue
    lab, el = LAB[k]; q = lab.any(axis=0)
    share = (q & rec[panel[k]]).sum() / max(q.sum(), 1)
    cat = "recurrent" if share >= 0.5 else "non-recurrent"
    for nm, P in (("blend", PB), ("NN", oN), ("LGBM", oL)):
        s = best_set(P[k].ravel(), np.ones(P[k].size, bool)).reshape(P[k].shape)
        res[(nm, cat)].append(iou(s, lab, el))
    n0, n5 = lab[0].sum(), lab[5].sum()
    g = "grow>50%" if n5 > 1.5 * max(n0, 1) else ("shrink>50%" if n5 < 0.5 * n0 else "stable")
    s = best_set(PB[k].ravel(), np.ones(PB[k].size, bool)).reshape(PB[k].shape)
    trend[g].append(iou(s, lab, el)); trend["cat:" + cat + ":" + g].append(iou(s, lab, el))
for nm in ("blend", "NN", "LGBM"):
    print(nm, {c: (len(res[(nm, c)]), round(float(np.mean(res[(nm, c)])), 3)) for c in ("recurrent", "non-recurrent")})
print({g: (len(v), round(float(np.mean(v)), 3)) for g, v in trend.items()})
