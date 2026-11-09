"""G99 (10/24, report Fig. 5): X1's gain on ongoing windows as a function of k, the number of completed earlier days the
fine-tune could use. M3 v3 ongoing windows outside the hold-out, e4s_a seeds 0, 1 (training-month networks; the final B's
base), per k (not the k expectation): IoU(X1 fine-tuned on days D-k..D-1) - IoU(saved model), far gate + decoder of the
M3 v2 gate with that k. Mean and window-bootstrap SE per k -> work/t2/g99_x1_k.csv.
usage: TFB_M3VER=v3 python g99_x1_k_curve.py"""
import os, sys, pickle, numpy as np, pandas as pd
os.environ.setdefault("TFB_M3VER", "v3")
sys.path.insert(0, r"C:/Users/user/tfb/src"); sys.argv = [sys.argv[0], "0"]
import t2_m3_gate as G, t2_m3v2_gate as V, m3v2

W = G.W; ld = lambda f: pickle.load(open(W + f, "rb")); av = lambda a, b: {k: (a[k] + b[k]) / 2 for k in a}
BASE = av(ld(f"probs_{G.PRE}__e4s_a_s0.pkl"), ld(f"probs_{G.PRE}__e4s_a_s1.pkl"))
X1 = av(ld(f"probs_m3{G.VER}v2x1__e4s_a_s0.pkl"), ld(f"probs_m3{G.VER}v2x1__e4s_a_s1.pkl"))
rows = []
for w in G.WID:
    m = G.META[w]
    if m["cond"] != "queue_ongoing" or m3v2.is_holdout(m["day"]):
        continue
    for k in range(1, min(6, m["day"]) + 1):
        sc = lambda p: G.iou(V.ongoing_pred(p, m, True, k), m)
        rows.append(dict(w=w, k=k, d=sc(X1[(w, k)]) - sc(BASE[w])))
R = pd.DataFrame(rows); rng = np.random.default_rng(0); out = []
for k, g in R.groupby("k"):
    d = g.d.values; bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(1000)]
    out.append(dict(k=k, n=len(d), mean=d.mean(), se=float(np.std(bs))))
O = pd.DataFrame(out); O.to_csv(W + "g99_x1_k.csv", index=False); print(O.round(4).to_string(index=False))
