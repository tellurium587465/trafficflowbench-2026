"""G95 (10/23, after LB s80q 0.89741 > s76 0.89608): on M3 v3, the ongoing blend of the R world (s70 spec: q43f + q46f, CF5
weights, seeds 0-3) vs the train-months q24f (CF5, seeds 0-3), saved models (no X1 / TTA), far gate + decoder, k expectation.
Does the instrument agree with the LB that the R world's synthetic ongoing is not better than the train months alone?
usage: TFB_M3VER=v3 python g95_family_m3.py"""
import os, sys, pickle
os.environ.setdefault("TFB_M3VER", "v3")
sys.path.insert(0, r"C:/Users/user/tfb/src"); sys.argv = [sys.argv[0], "0"]
import t2_m3_gate as G, t2_m3v2_gate as V
CF5 = (("attnglob", 2.0), ("attnwp05", 2.0), ("attn1", 2.0), ("drop1", 1.0), ("r13", 1.0))
FAM = {"R": [(f"{f}_{c}_s{s}", w) for s in range(4) for f in ("q43f", "q46f") for c, w in CF5],
       "q24f": [(f"q24f_{c}_s{s}", w) for s in range(4) for c, w in CF5]}
extra = os.environ.get("G95_FAM", "")                           # e.g. c24f once its M3 inference exists
if extra:
    FAM[extra] = [(f"{extra}_{c}_s{s}", w) for s in range(4) for c, w in CF5]
PR = {t: pickle.load(open(G.W + f"probs_{G.PRE}__{t}.pkl", "rb")) for sp in FAM.values() for t, _ in sp}
cols = {n: {} for n in FAM}
for w in G.WID:
    m = G.META[w]
    if m["cond"] != "queue_ongoing":
        continue
    for n, sp in FAM.items():
        cols[n][w] = V.score(sum(wt * PR[t][w] for t, wt in sp) / sum(wt for _, wt in sp), m)
V.frame(cols).dropna(subset=list(cols)).to_csv(G.W + f"g95_family_m3{G.VER}.csv", index=False); print("written")
