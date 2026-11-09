"""Does a logit recalibration p' = sigmoid(a*logit(p)+b) of a T2 blend help the expected-IoU decoder?
Averaging members flattens probabilities, and best_set picks the set size from their magnitudes, so (a, b) is chosen
per condition by nested CV over the three month folds (same objective as t2_blend_opt) and compared with a=1, b=0.

usage: python t2_calib.py <spec>
"""
import sys, numpy as np
import t2_eval_surprise as S
from t2_eval_oof import T, load_oof

FOLD = {**{m: 0 for m in ("2030-06", "2030-07", "2030-08")}, **{m: 1 for m in ("2030-09", "2030-10", "2030-11")},
        **{m: 2 for m in ("2030-12", "2031-01", "2031-02")}}


def iou(p, lab, el):                                         # same decoder/score as t2_blend_opt (which runs on import)
    pe = p.ravel(); order = np.argsort(-pe); cs = np.cumsum(pe[order]); kk = np.arange(1, pe.size + 1)
    ei = cs / (kk + pe.sum() - cs); kb = int(np.argmax(ei)) + 1; s = np.zeros(pe.size, bool); s[order[:kb]] = True
    s = s.reshape(p.shape); lab = lab.astype(bool); el = el.astype(bool); u = ((s | lab) & el).sum()
    return 1.0 if u == 0 else (s & lab & el).sum() / u


A = (0.7, 0.85, 1.0, 1.15, 1.3, 1.5); B = (-0.6, -0.3, -0.15, 0.0, 0.15, 0.3, 0.6)
grid = [(a, b) for a in A for b in B]; ident = grid.index((1.0, 0.0))
O = load_oof(sys.argv[1])

for cond in ("queue_onset", "queue_ongoing"):
    if cond == "queue_onset":
        ks = [k for k in S.SEL if k in O]; wsur = {k: w for k, w in zip(S.SEL, S.W1)}
    else:
        ks = [k for k in range(T.NWIN) if T.META["is_sel"][k] and T.META["cond"][k] == cond and k in O]
    fold = np.array([FOLD[str(T.META["date"][k])[:7]] for k in ks]); wv = np.array([wsur[k] for k in ks]) if cond == "queue_onset" else None
    R = np.zeros((len(grid), len(ks)))
    for j, k in enumerate(ks):
        p = np.clip(O[k], 1e-6, 1 - 1e-6); lg = np.log(p / (1 - p)); lab, el = T.LAB[k]
        for g, (a, b) in enumerate(grid):
            R[g, j] = iou(1 / (1 + np.exp(-(a * lg + b))), lab, el)

    def score(rows, idx):
        r = R[rows][:, idx] if np.ndim(rows) else R[rows, idx]
        if cond == "queue_onset":
            return 0.5 * r.mean(axis=-1) + 0.5 * (r * wv[idx]).sum(axis=-1) / wv[idx].sum()
        return r.mean(axis=-1)

    allg = np.arange(len(grid)); cv = []
    for f in range(3):
        te = np.flatnonzero(fold == f); gb = int(np.argmax(score(allg, np.flatnonzero(fold != f))))
        cv.append((score(gb, te) - score(ident, te), grid[gb]))
    gb = int(np.argmax(score(allg, np.arange(len(ks)))))
    print(f"[{cond}] windows {len(ks)} | identity {score(ident, np.arange(len(ks))):.4f} | nested-CV gain "
          f"{np.mean([c[0] for c in cv]):+.4f} per fold " + " ".join(f"{d:+.4f}@{g}" for d, g in cv) +
          f" | all-data best {grid[gb]} -> {score(gb, np.arange(len(ks))):.4f}", flush=True)
