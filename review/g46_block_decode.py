"""G46: block-consistent decoding for Task 2 (10/14). The generator computes the state on blocks of 2-9 links (adjacent
links inside a block are congested together: split rate ~0; twin/block_structure.py), so the label of a window is
block-constant (up to ineligible links). The decoder chooses links one by one; pooling the probabilities inside a block
(each step) is a structural prior that removes within-block disagreement.
Blocks are estimated per month fold from the OTHER folds' raw observations (split rate <= 0.05, pairs congested >= 30
rows), so no evaluated horizon enters its own block map. Members/weights = the no-memory blend of G42 (production-like),
windows/scoring = G42 (selector-like windows, onset mornings, onset scored at T+30 as the production decoder does).
Decoders (fixed a priori): D0 per link; D1 block mean; D2 half way (0.5 p + 0.5 block mean).
Also reports how often the labels themselves split a block (eligible links, positive steps).
usage: python g46_block_decode.py"""
import sys, numpy as np, pandas as pd
sys.path.insert(0, r"C:/Users/user/tfb/src"); sys.path.insert(0, r"C:/Users/user/tfb/twin")
from t2_eval_oof import T, oof_of
from t2_model import best_set
from cachelib import load, T2_PANELS
from onset_shape3_lib import raw_matrices

MEM = ["q25_nomem_s3", "q25_nomem_s4", "q31_nomem_wp05_s0", "q31_nomem_attn1_s0", "q31_nomem_r13_s0", "q31_nomem_drop1_s0"]
CUR = np.array([1, 1, 2, 2, 1, 1], float); CUR /= CUR.sum()
FOLD = {**{m: 0 for m in ("2030-06", "2030-07", "2030-08")}, **{m: 1 for m in ("2030-09", "2030-10", "2030-11")},
        **{m: 2 for m in ("2030-12", "2031-01", "2031-02")}}

BID = {}
for p in T2_PANELS:
    P = load(p, "train"); V, _, _ = raw_matrices(P)
    fold_row = np.array([FOLD.get(str(t)[:7], -1) for t in P.times])
    ok = np.isfinite(V); cong = (V <= P.vcut[None, :]) & ok
    for f in range(3):
        rows = fold_row != f
        a, b = cong[rows, :-1], cong[rows, 1:]; both = ok[rows, :-1] & ok[rows, 1:]
        n_any = ((a | b) & both).sum(0); split = ((a ^ b) & both).sum(0) / np.maximum(n_any, 1)
        tight = (n_any >= 30) & (split <= 0.05); bid = np.zeros(P.n, int)
        for i in range(1, P.n):
            bid[i] = bid[i - 1] if tight[i - 1] else bid[i - 1] + 1
        BID[p, f] = bid
    print(p, "blocks per fold", [int(BID[p, f].max() + 1) for f in range(3)], "of", P.n, "links", flush=True)
    load.cache_clear(); del V, ok, cong

O = [oof_of(m) for m in MEM]
keys = sorted(set.intersection(*[set(o) for o in O]) & {k for k in range(T.NWIN) if T.META["is_sel"][k]})


def pool(p, bid, lam):
    cnt = np.bincount(bid); m = np.stack([np.bincount(bid, weights=row) / cnt for row in p])[:, bid]
    return (1 - lam) * p + lam * m


def iou(p, y, e, cond):
    q = p.copy()
    if cond == "queue_onset":
        q[:5] = 0
    s = best_set(q.ravel(), np.ones(q.size, bool)).reshape(q.shape)
    u = ((s | y) & e).sum(); return 1.0 if u == 0 else (s & y & e).sum() / u


out, lab_split = [], []
for k in keys:
    cond = str(T.META["cond"][k]); t = int(T.META["t"][k])
    if cond == "queue_onset" and (t + 6) % 288 >= 144:
        continue
    pnl = str(T.META["panel"][k]); f = FOLD[str(T.META["date"][k])[:7]]; bid = BID[pnl, f]
    lab, el = T.LAB[k]; y = lab.astype(bool); e = el.astype(bool)
    pr = np.tensordot(CUR, np.stack([o[k] for o in O]), 1)
    row = dict(cond=cond, cl=pnl + str(T.META["date"][k])[:10])
    for nm, lam in (("D0", 0.0), ("D1", 1.0), ("D2", 0.5)):
        row[nm] = iou(pool(pr, bid, lam) if lam else pr, y, e, cond)
    out.append(row)
    for s in range(6):
        for blk in np.unique(bid[(y[s] & e[s])]):
            mem = (bid == blk) & e[s]
            if mem.sum() >= 2:
                lab_split.append(bool(y[s][mem].any() and not y[s][mem].all()))
D = pd.DataFrame(out); rng = np.random.default_rng(0)
print(f"labels: share of (positive block, step) with >= 2 eligible links that the label splits: {np.mean(lab_split):.3f} (n={len(lab_split)})")
for cond, g in D.groupby("cond"):
    u = g.cl.unique(); idx = {x: np.flatnonzero(g.cl.to_numpy() == x) for x in u}; msg = []
    for nm in ("D1", "D2"):
        d = (g[nm] - g["D0"]).to_numpy()
        se = np.std([d[np.concatenate([idx[x] for x in rng.choice(u, len(u))])].mean() for _ in range(500)])
        msg.append(f"{nm} {d.mean():+.4f} (SE {se:.4f})")
    print(f"{cond}: windows {len(g)}, D0 {g.D0.mean():.4f} | " + " | ".join(msg), flush=True)
