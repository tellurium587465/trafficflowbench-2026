"""Review: onset as a choice among each panel's top candidate sites. Accuracy by the true site's frequency rank,
and whether a location-free physics score picks the right candidate among the top sites."""
import os, sys, numpy as np
from collections import defaultdict, Counter
sys.path.insert(0, r"C:/Users/user/tfb/src")
os.environ.setdefault("TFB_T2CACHE", "2_noT")
import t2_eval_oof as E
from t2_model import best_set, FOLDS

M = E.T.META; LAB = E.T.LAB; cond = M["cond"]; panel = M["panel"]
oL = E.oof_of("l127_fast_af"); oN = E.oof_of("q12_af_r13_nopos")
keys = sorted(set(oL) & set(oN)); PB = {k: 0.5 * oL[k] + 0.5 * oN[k] for k in keys}
mon = np.array([d[:7] for d in M["date"]]); fold_of = {m: i for i, f in enumerate(FOLDS) for m in f}
allon = [k for k in range(E.T.NWIN) if cond[k] == "queue_onset"]
R = 3
sites = {}
for f in range(3):
    acc = {}
    for k in allon:
        if fold_of.get(mon[k], -1) == f: continue
        a = acc.setdefault(panel[k], [0, 0]); a[0] = a[0] + LAB[k][0][5].astype(float); a[1] += 1
    for p, (s, n) in acc.items():
        fr = np.convolve(s / n, np.ones(2 * R + 1), "same"); lst = []
        for _ in range(3):
            j = int(np.argmax(fr)); lst.append((j, float(fr[j]))); fr[max(0, j - 2 * R):j + 2 * R + 1] = -1
        sites[(f, p)] = lst


def site_of(f, p, j):
    for r, (c, _) in enumerate(sites[(f, p)]):
        if abs(j - c) <= R: return r
    return 3                                    # outside the top-3


iou = lambda s, lab, el: 1.0 if ((s | lab) & el).sum() == 0 else ((s & lab & el).sum()) / ((s | lab) & el).sum()
dec = lambda P: best_set(P.ravel(), np.ones(P.size, bool)).reshape(P.shape)
models = {"LGBM": oL, "NN": oN, "blend.5": PB, "blend.7": {k: 0.3 * oL[k] + 0.7 * oN[k] for k in keys}}
R_ = defaultdict(lambda: defaultdict(list))
for k in keys:
    if cond[k] != "queue_onset" or not LAB[k][0][5].any(): continue
    f = fold_of[mon[k]]; p = panel[k]; q = np.flatnonzero(LAB[k][0][5])
    ts = Counter(site_of(f, p, j) for j in q).most_common(1)[0][0]
    for nm_, P in models.items():
        R_[nm_][(p, ts)].append(iou(dec(P[k]), LAB[k][0], LAB[k][1]))
for nm_, d_ in R_.items():
    ps = sorted({p for p, _ in d_})
    nat = np.mean([np.mean(sum((d_[(p, r)] for r in range(4)), [])) for p in ps])
    flat = np.mean([np.mean([np.mean(d_[(p, r)]) for r in range(2) if d_[(p, r)]]) for p in ps])   # rank-1 and rank-2 sites equally likely
    oth = np.mean(sum((d_[(p, 3)] for p in ps), []))
    mix = np.mean([0.86 * np.mean([np.mean(d_[(p, r)]) for r in range(3) if d_[(p, r)]]) + 0.14 * oth for p in ps])
    print(f"{nm_:8s} natural {nat:.4f} | flat top-2 sites {flat:.4f} | other-site IoU {oth:.3f} | LB-like (top-3 flat 86% + other 14%) {mix:.4f}")
