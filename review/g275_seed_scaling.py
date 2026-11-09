"""G275 (11/05, ledger #233, declared by the strategy before the results; the user's instruction to decide the number of networks
statistically): how IoU grows with the number of seeds of the main networks. M3 v3 dev windows (hold-out days not read), saved
probabilities of q24f x 5 configurations (weights 2, 2, 2, 1, 1) over seeds 0-15; a subset of n seeds (n = 1, 2, 4, 8, 16; 50 random
draws each, 1 for 16) is averaged and decoded by the raw best_set (the networks alone: no evidence rule, no physics term, no X1), IoU by
condition (onset scored at T+30, ongoing over the horizon, M3's scored cells). Fit IoU(n) = a - c / n per condition on the means; the
predicted gains 16 -> 32 (c / 32) and 32 -> 64 (c / 64) with a window bootstrap SE (200). January / February: the fold-2 networks exist
for seeds 0-3 only, so not read. usage: TFB_M3VER=v3 py -B -X utf8 g275_seed_scaling.py"""
import os, sys, pickle
os.environ.setdefault("TFB_M3VER", "v3")
import numpy as np, pandas as pd
sys.path.insert(0, r"C:/Users/user/tfb/src")
import m3v2
from t2_model import best_set
W = "C:/Users/user/tfb/work/t2/"; CF5 = (("attnglob", 2.0), ("attnwp05", 2.0), ("attn1", 2.0), ("drop1", 1.0), ("r13", 1.0))
META = [m for m in pickle.load(open(W + "m3v3_meta.pkl", "rb"))["meta"] if not m3v2.is_holdout(np.array([m["day"]]))[0]]
P = {(c, s): pickle.load(open(W + f"probs_m3v3__q24f_{c}_s{s}.pkl", "rb")) for s in range(16) for c, _ in CF5}


def iou(sel, m):
    y, mk = m["y"].T.astype(bool), m["m"].T.astype(bool)
    if m["cond"] == "queue_onset":
        sel, y, mk = sel[-1:], y[-1:], mk[-1:]
    u = ((sel | y) & mk).sum(); return 1.0 if u == 0 else float((sel & y & mk).sum() / u)


rng = np.random.default_rng(0); NS = (1, 2, 4, 8, 16); rows = []
SUB = {n: ([tuple(range(16))] if n == 16 else [tuple(sorted(rng.choice(16, n, replace=False))) for _ in range(50)]) for n in NS}
for m in META:
    w = m["window_id"]; S = {s: sum(wt * P[(c, s)][w] for c, wt in CF5) / 8.0 for s in range(16)}
    for n in NS:
        v = [iou(best_set((sum(S[s] for s in sub) / n).ravel(), np.ones(S[0].size, bool)).reshape(S[0].shape), m) for sub in SUB[n]]
        rows.append(dict(w=w, cond=m["cond"], n=n, iou=float(np.mean(v)), sd=float(np.std(v))))
D = pd.DataFrame(rows); D.to_csv(W + "g275_seed_scaling.csv", index=False)


def fit(tab):                                                   # tab: n -> mean IoU; least squares of IoU = a - c / n
    x = np.array([1.0 / n for n in NS]); y = np.array([tab[n] for n in NS]); A = np.column_stack([np.ones_like(x), -x]); a, c = np.linalg.lstsq(A, y, rcond=None)[0]; return a, c


for cond, d in D.groupby("cond"):
    piv = d.pivot(index="w", columns="n", values="iou"); a, c = fit(piv.mean().to_dict()); bs = []
    for _ in range(200):
        ix = rng.integers(0, len(piv), len(piv)); bs.append(fit(piv.iloc[ix].mean().to_dict())[1])
    se = float(np.std(bs)); means = " ".join(f"n{n} {piv[n].mean():.4f}" for n in NS)
    print(f"{cond} ({len(piv)} windows): {means} | fit a {a:.4f} c {c:.4f} (SE {se:.4f}) | predicted 16->32 {c / 32:+.5f} (SE {se / 32:.5f}), 32->64 {c / 64:+.5f} | between-draw sd at n=8 {d[d.n == 8].sd.mean():.4f}", flush=True)
