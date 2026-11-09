"""E5 on M3 (INSTRUCTIONS_WORKER (vii), strategy session 10/22: seeds 4 -> 8 before s74): the NOMEM blend (q24f, 5
configurations, weights 2,2,2,1,1) averaged over seeds 0-3 (current) vs seeds 0-7. Needs probs_m3__q24f_<cfg>_s4..7.pkl
(t2_nn_cpu_infer.py on nn_test_m3 after the FINAL runs of queue_e5_q24f.txt). Post-processing as the M3 gate.
Reports raw / production-weighted differences and P(B>A) per condition; adopt when >= +2 SE or, as variance reduction
that cannot hurt in expectation, when not worse than -1 SE (pre-registered in STATE.md before running).
usage: python t2_m3_e5.py"""
import sys
sys.argv = [sys.argv[0], "0"]
import pickle, numpy as np, pandas as pd
import t2_m3_gate as G
from m3_stats import summary, fmt


def blend(seeds):
    parts = [(w, pickle.load(open(G.W + f"probs_m3__{fam}_s{s}.pkl", "rb"))) for fam, w in G.CFG5("q24f") for s in seeds]
    tw = sum(w for w, _ in parts)
    return {k: sum(w * pr[k] for w, pr in parts) / tw for k in parts[0][1]}


A, B = blend(range(4)), blend(range(8))
rows = []
for w in G.WID:
    m = G.META[w]
    pred = (lambda p_: G.onset_pred(p_, m, "cur")) if m["cond"] == "queue_onset" else (lambda p_: G.ongoing_pred(p_, m, True))
    rows.append(dict(cond=m["cond"], panel=m["panel"], day=m["day"], s4=G.iou(pred(A[w]), m), s8=G.iou(pred(B[w]), m)))
D = pd.DataFrame(rows); D.to_csv(G.W + "m3_e5.csv", index=False)
print("mean |p8 - p4| over cells:", round(float(np.mean([np.abs(B[w] - A[w]).mean() for w in G.WID])), 5))
for cond in ("queue_onset", "queue_ongoing"):
    print(fmt(f"{cond} 8 seeds vs 4 seeds", summary(D[D.cond == cond], "s4", "s8")))
print(fmt("T2 8 seeds vs 4 seeds", summary(D, "s4", "s8")))
