"""G47: does the onset model use the breakdown law? (10/14) Real episode starts are flow-driven beyond the clock: 82-98%
of starts fall in the top flow quintile of their (site, slot) (twin/episode_cause.py). In a test month the connectors are
redrawn, so which block breaks down should move with the flow at T relative to that block's usual flow at that time.
Stacked test (nested by month fold): z_k = (30-min flow at T, max over block k's links - mu) / sd, mu/sd of block k at
the same 15-min slot and day type over the free-flowing rows of the OTHER folds; blocks also from the other folds.
Audit 10/14: flows from row T-1 (the official history is [T-60, T)); penalty 1e-7 (1e-3 on the mean loss shrank g);
cluster-bootstrap SE of g; no hard/easy split (it mixed fold-specific block ids and conditioned on the outcome).
Recalibration fitted on the other folds' onset windows (eligible links, step T+30, the NN no-memory blend of G42):
  R0: logit p' = a + b logit p            R1: logit p' = a + b logit p + g z            (g fitted, not tuned)
then decoded as production (expected-IoU set at T+30). Reports onset IoU of R0/R1 (R1 - R0 isolates the flow law), the
fitted g per fold, and the same on the 'hard' windows whose true onset block is not the panel's most frequent onset
block (closest real proxy for a moved map). Cluster bootstrap (panel x day).
usage: python g47_hazard_stack.py"""
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
O = [oof_of(m) for m in MEM]
keys = sorted(set.intersection(*[set(o) for o in O]) & {k for k in range(T.NWIN) if T.META["is_sel"][k]})
ons = [k for k in keys if str(T.META["cond"][k]) == "queue_onset" and (int(T.META["t"][k]) + 6) % 288 < 144]
print("onset windows", len(ons), flush=True)

Z = {}                                                       # (k) -> z per link (n,), block ids
for p in T2_PANELS:
    kp = [k for k in ons if str(T.META["panel"][k]) == p]
    if not kp:
        continue
    P = load(p, "train"); V, Q, _ = raw_matrices(P); D = P.T // 288
    fold_row = np.array([FOLD.get(str(t)[:7], -1) for t in P.times])
    ok = np.isfinite(V); cong = (V <= P.vcut[None, :]) & ok
    c = np.vstack([np.zeros((1, P.n)), np.cumsum(np.nan_to_num(Q), axis=0)]); kq = np.vstack([np.zeros((1, P.n)), np.cumsum(np.isfinite(Q), axis=0)])
    ma = (c[6:] - c[:-6]) / np.maximum(kq[6:] - kq[:-6], 1); ma[(kq[6:] - kq[:-6]) < 4] = np.nan
    ma = np.vstack([np.full((5, P.n), np.nan), ma]).astype(np.float32); del c, kq
    slot = (np.arange(P.T) % 288) // 3; we = np.repeat(np.array([pd.Timestamp(str(x)).dayofweek >= 5 for x in P.day[:D]]), 288)[:P.T]
    for f in range(3):
        rows = fold_row != f
        a, b = cong[rows, :-1], cong[rows, 1:]; both = ok[rows, :-1] & ok[rows, 1:]
        n_any = ((a | b) & both).sum(0); split = ((a ^ b) & both).sum(0) / np.maximum(n_any, 1)
        tight = (n_any >= 30) & (split <= 0.05); bid = np.zeros(P.n, int)
        for i in range(1, P.n):
            bid[i] = bid[i - 1] if tight[i - 1] else bid[i - 1] + 1
        nb = bid.max() + 1
        fb = np.stack([np.nanmax(ma[:, bid == j], axis=1) for j in range(nb)], axis=1)          # (T, nb)
        cb = np.stack([cong[:, bid == j].any(axis=1) for j in range(nb)], axis=1)
        fbm = np.where(cb, np.nan, fb)                                                         # block free-flowing
        mu = np.full((2, 96, nb), np.nan); sd = np.full((2, 96, nb), np.nan)
        for w_ in (0, 1):
            for s in range(96):
                sel = rows & (we == w_) & (slot == s)
                if sel.sum() > 5:
                    mu[w_, s] = np.nanmean(fbm[sel], axis=0); sd[w_, s] = np.nanstd(fbm[sel], axis=0) + 1e-6
        for k in kp:
            if FOLD[str(T.META["date"][k])[:7]] != f:
                continue
            t = int(T.META["t"][k]); zb = (fb[t - 1] - mu[int(we[t - 1]), slot[t - 1]]) / sd[int(we[t - 1]), slot[t - 1]]   # history ends at T-1
            Z[k] = (np.nan_to_num(np.clip(zb, -5, 5))[bid], bid)
    print(p, "windows with z", sum(1 for k in kp if k in Z), flush=True)
    load.cache_clear(); del V, Q, ma, ok, cong

rec = []
for k in ons:
    if k not in Z:
        continue
    lab, el = T.LAB[k]; y = lab.astype(bool); e = el.astype(bool)
    pr = np.clip(np.tensordot(CUR, np.stack([o[k] for o in O]), 1), 1e-5, 1 - 1e-5)
    rec.append(dict(k=k, f=FOLD[str(T.META["date"][k])[:7]], pnl=str(T.META["panel"][k]), cl=str(T.META["panel"][k]) + str(T.META["date"][k])[:10],
                    p=pr, y=y, e=e, z=Z[k][0], bid=Z[k][1]))
# most frequent onset block per panel (from all windows' labels: structure only, used to tag 'hard' windows)
mode_blk = {}
for pnl in {r["pnl"] for r in rec}:
    cnt = {}
    for r in rec:
        if r["pnl"] == pnl:
            for j in np.unique(r["bid"][r["y"][5] & r["e"][5]]):
                cnt[j] = cnt.get(j, 0) + 1
    mode_blk[pnl] = max(cnt, key=cnt.get) if cnt else -1


def design(rs, use_z):
    X, Y = [], []
    for r in rs:
        m = r["e"][5]; lp = np.log(r["p"][5][m] / (1 - r["p"][5][m]))
        X.append(np.column_stack([np.ones(m.sum()), lp] + ([r["z"][m]] if use_z else []))); Y.append(r["y"][5][m].astype(float))
    return np.vstack(X), np.concatenate(Y)


def fit(A, y):
    """mean log-loss + 1e-3 |w|^2 (the NN is nearly separating, plain IRLS diverges), L-BFGS"""
    from scipy.optimize import minimize
    def f(w):
        z = A @ w; nll = np.mean(np.logaddexp(0, z) - y * z) + 1e-7 * (w @ w)
        g = A.T @ (1 / (1 + np.exp(-z)) - y) / len(y) + 2e-7 * w
        return nll, g
    w0 = np.zeros(A.shape[1]); w0[1] = 1.0
    return minimize(f, w0, jac=True, method="L-BFGS-B").x


def iou(q, r):
    qq = np.zeros_like(r["p"]); qq[5] = q
    s = best_set(qq.ravel(), np.ones(qq.size, bool)).reshape(qq.shape)
    u = ((s | r["y"]) & r["e"]).sum(); return 1.0 if u == 0 else (s & r["y"] & r["e"]).sum() / u


out = []
for f in range(3):
    tr = [r for r in rec if r["f"] != f]; te = [r for r in rec if r["f"] == f]
    w0 = fit(*design(tr, False)); w1 = fit(*design(tr, True))
    cls = sorted({r["cl"] for r in tr}); bs_g = []
    for b_ in range(60):                                       # cluster bootstrap (panel x day) of the fitted g
        sub = [r for c_ in np.random.default_rng(b_).choice(cls, len(cls)) for r in tr if r["cl"] == c_]
        bs_g.append(fit(*design(sub, True))[2])
    print(f"fold {f}: g = {w1[2]:.3f} (cluster-bootstrap SE {np.std(bs_g):.3f})", flush=True)
    print(f"fold {f}: R0 a,b = {np.round(w0, 3).tolist()}  R1 a,b,g = {np.round(w1, 3).tolist()}", flush=True)
    for r in te:
        lp = np.log(r["p"][5] / (1 - r["p"][5]))
        q0 = 1 / (1 + np.exp(-(w0[0] + w0[1] * lp))); q1 = 1 / (1 + np.exp(-(w1[0] + w1[1] * lp + w1[2] * r["z"])))
        hard = not bool(((r["bid"] == mode_blk[r["pnl"]]) & r["y"][5] & r["e"][5]).any())
        out.append(dict(cl=r["cl"], hard=hard, raw=iou(r["p"][5], r), R0=iou(q0, r), R1=iou(q1, r)))
D = pd.DataFrame(out); rng = np.random.default_rng(0)
for nm, g in (("all", D),):
    u = g.cl.unique(); idx = {x: np.flatnonzero(g.cl.to_numpy() == x) for x in u}; msg = []
    for a, b in (("R0", "raw"), ("R1", "R0")):
        d = (g[a] - g[b]).to_numpy()
        se = np.std([d[np.concatenate([idx[x] for x in rng.choice(u, len(u))])].mean() for _ in range(500)])
        msg.append(f"{a}-{b} {d.mean():+.4f} (SE {se:.4f})")
    print(f"{nm}: windows {len(g)}, raw {g.raw.mean():.4f}, R0 {g.R0.mean():.4f}, R1 {g.R1.mean():.4f} | " + " | ".join(msg), flush=True)
