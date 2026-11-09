"""Feature fixes and input variants on M3 v2 (10/22 night chains; judged 10/23-24, strategy §(b) / §16.5):
  semp  = s_emp per segment +-25 km/h (probs_m3__semp_attnglob_s{0,1}, inferred on nn_test_m3_semp)
  arr5  = arr5 in ZERO               (probs_m3__arr5_attnglob_s{0,1}, nn_test_m3)
  pct   = within-window percentile inputs (probs_m3__pct_attnglob_s{0,1}, nn_test_m3_pct)
each vs the same configuration on the current inputs, q24f_attnglob seeds 0,1 (same seeds) and seeds 0-3 (more stable
base), single configuration, both conditions, t2_m3v2_gate.score (k expectation). Dev windows only (hold-out excluded);
model A/B -> z also with the A/A factor 1.25 (10/22). Variants whose probabilities are missing are skipped.
usage: python g63_feat_m3.py"""
import sys, os, pickle
sys.path.insert(0, r"C:/Users/user/tfb/src")
sys.argv = [sys.argv[0], "0"]
import numpy as np
import t2_m3_gate as G, t2_m3v2_gate as V, m3v2

ld = lambda t: pickle.load(open(G.W + f"probs_m3__{t}.pkl", "rb"))
avg = lambda ts: (lambda P: {w: sum(p[w] for p in P) / len(P) for w in P[0]})([ld(t) for t in ts])
cols = {"b01": avg([f"q24f_attnglob_s{s}" for s in (0, 1)]), "b0123": avg([f"q24f_attnglob_s{s}" for s in range(4)]),
        "b8": avg([f"q24f_attnglob_s{s}" for s in range(8)])}                # 10/23: + E5 seeds (no systematic env drift)
for v in ("semp", "arr5", "pct"):
    ts = [f"{v}_attnglob_s{s}" for s in (0, 1)]
    if all(os.path.exists(G.W + f"probs_m3__{t}.pkl") for t in ts):
        cols[v] = avg(ts)
S = {c: {w: V.score(p[w], G.META[w]) for w in G.WID} for c, p in cols.items()}
D = V.frame(S); D.to_csv(G.W + "m3v2_feat.csv", index=False); dev = D[~m3v2.is_holdout(D.day)]
for v in [c for c in cols if c not in ("b01", "b0123", "b8")]:
    for b in ("b01", "b0123", "b8"):
        s = m3v2.summary(dev, b, v, "private")
        print(m3v2.fmt(f"{v} vs {b} (dev)", s), f"| z x1.25: {s['t2_weighted'] / (1.25 * s['se']):+.2f}")
