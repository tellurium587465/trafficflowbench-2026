"""E5 on M3 v2 (10/22; t2_m3_e5.py under the v2 instrument): the onset production blend (q24f, 5 configurations, weights
2,2,2,1,1 as G.CFG5) averaged over seeds 0-3 vs seeds 0-7, scored with t2_m3v2_gate.score (k expectation, simulated
blanks) for both conditions. Judged on the windows outside the hold-out (m3v2.is_holdout), all windows for reference.
Seed count is variance reduction: pre-registered rule (STATE 10/22) adopt unless worse than -1 SE; the SE here needs no
seed-noise term for the direction (more seeds cannot be worse in expectation) but is printed x1.25 (A/A, 10/22) too.
usage: python t2_m3v2_e5.py"""
import sys
sys.argv = [sys.argv[0], "0"]
import pickle, numpy as np
import t2_m3_gate as G, t2_m3v2_gate as V, m3v2


def blend(seeds):
    parts = [(w, pickle.load(open(G.W + f"probs_m3__{fam}_s{s}.pkl", "rb"))) for fam, w in G.CFG5("q24f") for s in seeds]
    tw = sum(w for w, _ in parts)
    return {k: sum(w * pr[k] for w, pr in parts) / tw for k in parts[0][1]}


A, B = blend(range(4)), blend(range(8))
D = V.frame({"s4": {w: V.score(A[w], G.META[w]) for w in G.WID}, "s8": {w: V.score(B[w], G.META[w]) for w in G.WID}})
D.to_csv(G.W + "m3v2_e5.csv", index=False)
print("mean |p8 - p4| over cells:", round(float(np.mean([np.abs(B[w] - A[w]).mean() for w in G.WID])), 5))
dev = D[~m3v2.is_holdout(D.day)]
s = m3v2.summary(dev, "s4", "s8", "private")
print(m3v2.fmt("8 seeds vs 4 seeds, dev", s), f"| z with the A/A factor 1.25: {s['t2_weighted'] / (1.25 * s['se']):+.2f}")
print(m3v2.fmt("8 seeds vs 4 seeds, all windows (reference)", m3v2.summary(D, "s4", "s8", "private")))
