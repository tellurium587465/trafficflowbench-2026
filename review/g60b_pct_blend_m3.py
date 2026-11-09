"""Strategy bet 2 (i) (10/22): the no-memory LightGBM (t2_lgb_nomem_final.py) mixed into the production NN blends on M3 v2.
NN = onset s46 spec (q24f) / ongoing s70 spec (q43f/q46f) on nn_test_m3; mix = (1 - a) NN + a LGB per cell, a = 0.2
pre-declared (METAGAME_REPLY §4), 0.1 / 0.3 printed as the curve only. Judged on the windows outside the hold-out
(m3v2.is_holdout); all-window numbers printed for reference. Also the error correlation of NN and LGB per condition.
usage: python g60b_pct_blend_m3.py   (10/23: the percentile-input NN of strategy 16.5 instead of LightGBM)"""
import sys, json, pickle, numpy as np
sys.path.insert(0, r"C:/Users/user/tfb/src")
sys.argv = [sys.argv[0], "0"]
import t2_m3_gate as G, t2_m3v2_gate as V, m3v2

SPEC = {"queue_onset": json.load(open("C:/Users/user/tfb/src/t2_specs.json"))["s37"]["spec"],
        "queue_ongoing": "+".join(f"{f}_{c}_s{s}{w}" for s in range(4) for f in ("q43f", "q46f")
                                  for c, w in (("attnglob", ":2"), ("attnwp05", ":2"), ("attn1", ":2"), ("drop1", ""), ("r13", "")))}
_C = {}


def probs(tag):
    if tag not in _C:
        _C[tag] = pickle.load(open(G.W + f"probs_m3__{tag}.pkl", "rb"))
    return _C[tag]


def nn(c, w):
    parts = [(x.split(":")[0], float(x.split(":")[1]) if ":" in x else 1.0) for x in SPEC[c].split("+")]
    return sum(wt * probs(m)[w] for m, wt in parts) / sum(wt for _, wt in parts)


PCT = (lambda a, b: {w: (a[w] + b[w]) / 2 for w in a})(probs("pct_attnglob_s0"), probs("pct_attnglob_s1"))
A = (0.1, 0.2, 0.3); cols = {"nn": {}} | {f"a{a}": {} for a in A}; err = {"queue_onset": ([], []), "queue_ongoing": ([], [])}
for w in G.WID:
    m = G.META[w]; pn = nn(m["cond"], w); pl = PCT[w]
    cols["nn"][w] = V.score(pn, m)
    for a in A:
        cols[f"a{a}"][w] = V.score((1 - a) * pn + a * pl, m)
    y, ok = m["y"].T.astype(float), m["m"].T.astype(bool)            # meta: (n, 6); probabilities: (6, n)
    if m["cond"] == "queue_onset":
        ok = ok & (np.arange(6)[:, None] == 5)                       # onset predicts the last step only
    err[m["cond"]][0].append((pn - y)[ok]); err[m["cond"]][1].append((pl - y)[ok])
D = V.frame(cols); D.to_csv(G.W + "m3v2_pctblend.csv", index=False)
dev = D[~m3v2.is_holdout(D.day)]
for c, (a_, b_) in err.items():
    print(f"{c}: error correlation NN vs pct-NN {np.corrcoef(np.concatenate(a_), np.concatenate(b_))[0, 1]:.3f}")
for a in A:
    tag = " (pre-declared)" if a == 0.2 else " (curve only)"
    print(m3v2.fmt(f"NN -> mix a={a}{tag}, dev", m3v2.summary(dev, "nn", f"a{a}", "private")))
print(m3v2.fmt("NN -> mix a=0.2, all windows (reference)", m3v2.summary(D, "nn", "a0.2", "private")))
for c in ("queue_onset", "queue_ongoing"):
    g = dev[dev.cond == c]
    print(m3v2.fmt(f"{c} only, NN -> mix a=0.2 (dev)", m3v2.summary(g, "nn", "a0.2", "private")))
