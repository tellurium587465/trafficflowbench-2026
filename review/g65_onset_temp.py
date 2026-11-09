"""G65 (10/22 night): onset over-confidence in a new month. The production onset blend (s46 spec) gives p ~ 0.999 at the
memorised site even when a rare site breaks down (G64), so the expected-IoU set never hedges. Temperature on the onset
T+30 logits, p' = sigmoid(logit(p) / tau), before the evidence rule and the set decoder (t2_m3v2_gate.score, k
expectation). M3 dev onset windows only (hold-out excluded); tau chosen on one half of the dev days (day parity) and
scored on the other (2-fold cross-fit), plus the full curve. Ongoing untouched (T2 = onset part only, x 0.5 in T2).
usage: python g65_onset_temp.py"""
import sys, json, pickle, numpy as np, pandas as pd
sys.path.insert(0, r"C:/Users/user/tfb/src")
sys.argv = [sys.argv[0], "0"]
import t2_m3_gate as G, t2_m3v2_gate as V, m3v2
lg = lambda x: np.log(np.clip(x, 1e-7, 1 - 1e-7) / (1 - np.clip(x, 1e-7, 1 - 1e-7)))
spec = json.load(open(r"C:/Users/user/tfb/src/t2_specs.json"))["s37"]["spec"]
parts = [(x.split(":")[0], float(x.split(":")[1]) if ":" in x else 1.0) for x in spec.split("+")]
PR = {t: pickle.load(open(G.W + f"probs_m3__{t}.pkl", "rb")) for t, _ in parts}
TAUS = (1.0, 1.25, 1.5, 2.0, 3.0, 4.0)
WON = [w for w in G.WID if G.META[w]["cond"] == "queue_onset" and not m3v2.is_holdout(G.META[w]["day"])]
cols = {f"t{t}": {} for t in TAUS}
for w in WON:
    m = G.META[w]; p = sum(wt * PR[t][w] for t, wt in parts) / sum(wt for _, wt in parts)
    for t in TAUS:
        q = p.copy(); q[5] = 1 / (1 + np.exp(-lg(p[5]) / t)); cols[f"t{t}"][w] = V.score(q, m)
D = V.frame(cols).dropna(subset=[f"t{t}" for t in TAUS]); D.to_csv(G.W + "g65_onset_temp.csv", index=False)
for t in TAUS[1:]:
    print(m3v2.fmt(f"onset tau {t} vs 1.0 (dev, curve)", m3v2.summary(D, "t1.0", f"t{t}", "private")))
cf = []
for half in (0, 1):
    fit, ev = D[D.day % 2 == half], D[D.day % 2 != half]
    best = max(TAUS, key=lambda t: fit[f"t{t}"].mean()); cf.append((half, best, float(ev[f"t{best}"].mean() - ev["t1.0"].mean()), len(ev)))
print("cross-fit (fit on one day parity, score on the other):", [(h, b, round(d, 4), n) for h, b, d, n in cf])
