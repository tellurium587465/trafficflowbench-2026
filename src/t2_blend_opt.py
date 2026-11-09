"""Principled T2 blend weights (not by eye): per condition, search the weight simplex over component OOFs.

Objective (selector-like train windows, the official-like decoder best_set):
  onset   = 0.5 * mean IoU + 0.5 * calendar-surprise-weighted IoU (t2_eval_surprise weights)
  ongoing = mean IoU
Nested CV over the three month folds: weights are chosen on two folds and scored on the third, so the reported
'cv' number is honest; the final weights are chosen on all windows. Compared with equal weights.

usage: python t2_blend_opt.py <tag1> <tag2> [...]   (2-4 components; grid step 0.05)
"""
import os, sys, itertools, numpy as np
import t2_eval_surprise as S
from t2_eval_oof import T, load_oof

FOLD = {**{m: 0 for m in ("2030-06", "2030-07", "2030-08")}, **{m: 1 for m in ("2030-09", "2030-10", "2030-11")},
        **{m: 2 for m in ("2030-12", "2031-01", "2031-02")}}
tags = sys.argv[1:]; O = [load_oof(t) for t in tags]   # a component may be a spec "a:2+b"
STEP = 0.05 if len(tags) <= 3 else 0.1
grid = [w for w in itertools.product(np.arange(0, 1 + 1e-9, STEP), repeat=len(tags)) if abs(sum(w) - 1) < 1e-6]


def iou(p, lab, el):
    pe = p.ravel(); order = np.argsort(-pe); cs = np.cumsum(pe[order]); kk = np.arange(1, pe.size + 1)
    ei = cs / (kk + pe.sum() - cs); kb = int(np.argmax(ei)) + 1; s = np.zeros(pe.size, bool); s[order[:kb]] = True
    s = s.reshape(p.shape); lab = lab.astype(bool); el = el.astype(bool); u = ((s | lab) & el).sum()
    return 1.0 if u == 0 else (s & lab & el).sum() / u


for cond in ("queue_onset", "queue_ongoing"):
    if cond == "queue_onset":
        ks = [k for k in S.SEL if all(k in o for o in O)]; wsur = {k: w for k, w in zip(S.SEL, S.W1)}
    else:
        ks = [k for k in range(T.NWIN) if T.META["is_sel"][k] and T.META["cond"][k] == cond and all(k in o for o in O)]
    fold = np.array([FOLD[str(T.META["date"][k])[:7]] for k in ks]); wv = np.array([wsur[k] for k in ks]) if cond == "queue_onset" else None
    R = np.zeros((len(grid), len(ks)))                     # per-window IoU for every weight vector
    for j, k in enumerate(ks):
        P = [o[k] for o in O]; lab, el = T.LAB[k]
        for g, w in enumerate(grid):
            R[g, j] = iou(sum(wi * p for wi, p in zip(w, P)), lab, el)

    def score(rows, idx):
        r = R[rows][:, idx] if np.ndim(rows) else R[rows, idx]
        if cond == "queue_onset":
            return 0.5 * r.mean(axis=-1) + 0.5 * (r * wv[idx]).sum(axis=-1) / wv[idx].sum()
        return r.mean(axis=-1)

    allg = np.arange(len(grid)); base = np.array([float(x) for x in os.environ["BASE"].split(",")]) if os.environ.get("BASE") else np.full(len(tags), 1 / len(tags))
    eq = int(np.argmin([np.abs(np.array(w) - base / base.sum()).sum() for w in grid]))   # BASE=0.2,0.6,0.1,0.1: compare with a given blend
    cv_best, cv_eq = [], []
    for f in range(3):
        tr, te = fold != f, fold == f
        gb = int(np.argmax(score(allg, np.flatnonzero(tr))))
        cv_best.append(score(gb, np.flatnonzero(te))); cv_eq.append(score(eq, np.flatnonzero(te)))
    gb = int(np.argmax(score(allg, np.arange(len(ks)))))
    print(f"[{cond}] windows {len(ks)} | nested-CV: chosen {np.mean(cv_best):.4f} vs equal {np.mean(cv_eq):.4f} "
          f"(diff {np.mean(cv_best) - np.mean(cv_eq):+.4f}) | final weights " +
          ", ".join(f"{t}={w:.2f}" for t, w in zip(tags, grid[gb])) + f" -> {score(gb, np.arange(len(ks))):.4f}", flush=True)
    print("    per-fold chosen-base " + " ".join(f"{b - e:+.4f}" for b, e in zip(cv_best, cv_eq)) + f" | base weights {grid[eq]}")
    for t, i in zip(tags, range(len(tags))):
        solo = [g for g, w in enumerate(grid) if w[i] > 0.999]
        if solo:
            print(f"    solo {t}: {score(solo[0], np.arange(len(ks))):.4f}")
