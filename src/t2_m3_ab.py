"""Generic M3 A/B of two model sets (10/22): each set's probabilities (probs_m3__<tag>.pkl, t2_nn_cpu_infer.py on
nn_test_m3) averaged with equal weights, the M3 gate's post-processing (current evidence rule, far gate, k-day limit,
trap 4), raw / production-weighted differences B - A, cluster SE, P(B>A) over 80-window redraws, per condition.
usage: python t2_m3_ab.py <tagA[,tagA...]> <tagB[,tagB...]> [name]"""
import sys
A_TAGS, B_TAGS = sys.argv[1].split(","), sys.argv[2].split(","); NAME = sys.argv[3] if len(sys.argv) > 3 else "ab"
sys.argv = [sys.argv[0], "0"]
import pickle, numpy as np, pandas as pd
import t2_m3_gate as G
from m3_stats import summary, fmt


def mean_probs(tags):
    ps = [pickle.load(open(G.W + f"probs_m3__{t}.pkl", "rb")) for t in tags]
    return {k: sum(p[k] for p in ps) / len(ps) for k in ps[0]}


A, B = mean_probs(A_TAGS), mean_probs(B_TAGS)
rows = []
for w in G.WID:
    m = G.META[w]
    pred = (lambda p_: G.onset_pred(p_, m, "cur")) if m["cond"] == "queue_onset" else (lambda p_: G.ongoing_pred(p_, m, True))
    rows.append(dict(cond=m["cond"], panel=m["panel"], day=m["day"], a=G.iou(pred(A[w]), m), b=G.iou(pred(B[w]), m)))
D = pd.DataFrame(rows); D.to_csv(G.W + f"m3_ab_{NAME}.csv", index=False)
print(f"{NAME}: A = {A_TAGS} | B = {B_TAGS} | mean |pB - pA| {np.mean([np.abs(B[w] - A[w]).mean() for w in G.WID]):.5f}")
for cond in ("queue_onset", "queue_ongoing"):
    print(fmt(f"{cond} B - A", summary(D[D.cond == cond], "a", "b")))
print(fmt("T2 B - A", summary(D, "a", "b")))
