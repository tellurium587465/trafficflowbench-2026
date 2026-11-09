"""Score the q_today training-filter input (g59_qtoday_m3.py) on M3 v2: the production blends (onset = s46 spec on q24f,
ongoing = s70 spec on q43f/q46f) on nn_test_m3 (probs_m3__) vs nn_test_m3q (probs_m3q__). Same models, only the input
column differs -> no seed-noise term. Also reports the result per history-coverage band (the M3 histories are ~0.57
observed, official ~0.76, so count features act differently; review B's warning).
usage: python g59_qtoday_score.py"""
import sys, json, pickle, numpy as np
sys.path.insert(0, r"C:/Users/user/tfb/src")
sys.argv = [sys.argv[0], "0"]
import t2_m3_gate as G, t2_m3v2_gate as V, m3v2

SPEC = {"queue_onset": json.load(open("C:/Users/user/tfb/src/t2_specs.json"))["s37"]["spec"],
        "queue_ongoing": "+".join(f"{f}_{c}_s{s}{w}" for s in range(4) for f in ("q43f", "q46f")
                                  for c, w in (("attnglob", ":2"), ("attnwp05", ":2"), ("attn1", ":2"), ("drop1", ""), ("r13", "")))}
_C = {}


def probs(pre, tag):
    if (pre, tag) not in _C:
        _C[(pre, tag)] = pickle.load(open(G.W + f"probs_{pre}__{tag}.pkl", "rb"))
    return _C[(pre, tag)]


def blend(pre, c, w):
    parts = [(x.split(":")[0], float(x.split(":")[1]) if ":" in x else 1.0) for x in SPEC[c].split("+")]
    return sum(wt * probs(pre, m)[w] for m, wt in parts) / sum(wt for _, wt in parts)


cols = {"base": {}, "qfix": {}}
for w in G.WID:
    m = G.META[w]
    for name, pre in (("base", "m3"), ("qfix", "m3q")):
        cols[name][w] = V.score(blend(pre, m["cond"], w), m)
D = V.frame(cols); D["cov"] = [G.META[w]["cov"] for w in D.w]
D.to_csv(G.W + "m3v2_qtoday.csv", index=False)
print(m3v2.fmt("q_today training filter (T2)", m3v2.summary(D, "base", "qfix", "private")))
print(m3v2.fmt("  (March composition)", m3v2.summary(D, "base", "qfix", "validation")))
for lo, hi in ((0.5, 0.55), (0.55, 0.6), (0.6, 1.01)):
    g = D[(D["cov"] >= lo) & (D["cov"] < hi)]
    print(f"  coverage {lo:.2f}-{hi:.2f}: n {len(g)}, onset {g[g.cond == 'queue_onset'].eval('qfix - base').mean():+.4f}, "
          f"ongoing {g[g.cond == 'queue_ongoing'].eval('qfix - base').mean():+.4f}")
