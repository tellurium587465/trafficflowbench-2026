"""G98 (10/23, the report's reproducibility section; strategy 22:05): how far does a RETRAINED network (same config line,
GPU / ROCm, not bit-exact) move the outputs? e4f_attnglob_s1rep and q24f_attnglob_s1rep vs the saved e4f_attnglob_s1 /
q24f_attnglob_s1. (1) official windows: |p_rep - p_saved| on eligible-size arrays; (2) M3 v3 (train-months model only):
the final B's blends (onset q24f s0-7, ongoing q24f s0-3, saved models, no X1) with the one network swapped -> paired T2
difference (S-equivalent x 0.30) via src/judge.py; (3) the final A's April queue with e4f_attnglob_s1 swapped (base and X1)
-> cells changed (g98_queue_A_rep, built separately).
usage: TFB_M3VER=v3 python g98_rep_variance.py"""
import os, sys, pickle, numpy as np
os.environ.setdefault("TFB_M3VER", "v3")
sys.path.insert(0, r"C:/Users/user/tfb/src"); sys.argv = [sys.argv[0], "0"]
import t2_m3_gate as G, t2_m3v2_gate as V

W = G.W; ld = lambda f: pickle.load(open(W + f, "rb"))
for tag in ("e4f_attnglob_s1", "q24f_attnglob_s1"):
    a, b = ld(f"probs_{tag}.pkl"), ld(f"probs_{tag}rep.pkl"); d = np.concatenate([np.abs(a[w] - b[w]).ravel() for w in a])
    print(f"{tag}: official windows |p_rep - p_saved| mean {d.mean():.4f}, p95 {np.percentile(d, 95):.4f}, max {d.max():.4f}")
CF5 = (("attnglob", 2.0), ("attnwp05", 2.0), ("attn1", 2.0), ("drop1", 1.0), ("r13", 1.0))
ON = [(f"q24f_{c}_s{s}", w) for s in range(8) for c, w in CF5]; OG = [(f"q24f_{c}_s{s}", w) for s in range(4) for c, w in CF5]
PR = {t: ld(f"probs_{G.PRE}__{t}.pkl") for t, _ in ON}; PR["q24f_attnglob_s1rep"] = ld(f"probs_{G.PRE}__q24f_attnglob_s1rep.pkl")
blend = lambda sp, w, rep: sum(wt * PR[t + ("rep" if rep and t == "q24f_attnglob_s1" else "")][w] for t, wt in sp) / sum(wt for _, wt in sp)
cols = {"saved": {}, "rep": {}}
for w in G.WID:
    m = G.META[w]; sp = ON if m["cond"] == "queue_onset" else OG
    for c, rep in (("saved", False), ("rep", True)):
        cols[c][w] = V.score(blend(sp, w, rep), m)
V.frame(cols).dropna(subset=list(cols)).to_csv(W + f"g98_rep_m3{G.VER}.csv", index=False); print("M3 frame written")
