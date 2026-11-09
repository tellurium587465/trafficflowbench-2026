"""G75 (10/23, X1 overfit fix "alpha(k)", NEXT_PLAN 10/25-27 pulled forward): mix the fine-tuned and the saved model by the
number of earlier days k the fine-tune saw: p = (1 - alpha_k) p_base + alpha_k p_X1(k), alpha per k group (k = 1-2, 3-4,
5-6), from the existing M3 v2 X1 outputs (probs_m3v2x1__e4s_a_s{0,1}, keyed (window, k); base = probs_m3__e4s_a). Scored
with t2_m3v2_gate (k expectation; k = 0 keeps the base), per condition, dev windows only (hold-out excluded).
alpha grid {0, 0.25, 0.5, 0.75, 1} per group, chosen by coordinate search on one day-parity half and scored on the other
(2-fold cross-fit) against alpha = 1 (the current X1) and alpha = 0 (base).
usage: python g75_x1_alpha_k.py"""
import sys, pickle, itertools
sys.path.insert(0, r"C:/Users/user/tfb/src")
sys.argv = [sys.argv[0], "0"]
import numpy as np, pandas as pd
import t2_m3_gate as G, t2_m3v2_gate as V, m3v2

B = [pickle.load(open(G.W + f"probs_m3__e4s_a_s{s}.pkl", "rb")) for s in (0, 1)]
X = [pickle.load(open(G.W + f"probs_m3v2x1__e4s_a_s{s}.pkl", "rb")) for s in (0, 1)]
GRID = (0.0, 0.25, 0.5, 0.75, 1.0); GROUPS = {1: 0, 2: 0, 3: 1, 4: 1, 5: 2, 6: 2}
WD = [w for w in G.WID if not m3v2.is_holdout(G.META[w]["day"])]
# per window, per k, per alpha: the IoU (k expectation is applied when combining the groups)
S = {}
for w in WD:
    m = G.META[w]; base = (B[0][w] + B[1][w]) / 2
    for k, pk in m3v2.kdist(m["day"]).items():
        xk = (X[0][(w, k)] + X[1][(w, k)]) / 2 if k > 0 else base
        for a in GRID:
            p = (1 - a) * base + a * xk if k > 0 else base
            S[(w, k, a)] = (pk, G.iou(V.onset_pred(p, m, "cur", k) if m["cond"] == "queue_onset" else V.ongoing_pred(p, m, True, k), m))


def score(alphas, ws):
    """alphas: per group (3,) -> {window: expected IoU}"""
    out = {}
    for w in ws:
        m = G.META[w]; out[w] = sum(S[(w, k, alphas[GROUPS[k]] if k > 0 else 0.0)][0] * S[(w, k, alphas[GROUPS[k]] if k > 0 else 0.0)][1]
                                   for k in m3v2.kdist(m["day"]))
    return out


for cond in ("queue_ongoing", "queue_onset"):
    ws = [w for w in WD if G.META[w]["cond"] == cond]
    cols = {"base": score((0, 0, 0), ws), "x1": score((1, 1, 1), ws)}
    best_all = max(itertools.product(GRID, repeat=3), key=lambda a: np.mean(list(score(a, ws).values())))
    cols["fit_all"] = score(best_all, ws)
    cf = {}
    for half in (0, 1):
        fit = [w for w in ws if G.META[w]["day"] % 2 == half]; ev = [w for w in ws if G.META[w]["day"] % 2 != half]
        a = max(itertools.product(GRID, repeat=3), key=lambda a: np.mean(list(score(a, fit).values())))
        cf.update(score(a, ev)); print(f"{cond}: fitted on parity {half}: alpha(k=1-2, 3-4, 5-6) = {a}")
    cols["crossfit"] = cf
    D = V.frame(cols).dropna(subset=list(cols))
    print(f"{cond}: in-sample best alpha {best_all} (optimistic)")
    for a_, b_ in (("base", "x1"), ("x1", "crossfit"), ("base", "crossfit")):
        print(m3v2.fmt(f"  {cond} {a_} -> {b_} (dev)", m3v2.summary(D, a_, b_, "private")))
