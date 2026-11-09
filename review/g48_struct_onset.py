"""G48: a structural onset law that should transfer to a redrawn month (10/14). Measured facts: the generator switches a
whole block to the congested FD branch when the demand reaches the bottleneck's discharge flow; that discharge flow is
the same in the train, validation and private months for every block (twin/capacity_shift.py: |d q/cap| 0.003-0.036),
the demand's time-of-day shape is unchanged, and the connectors (the flows) are redrawn. So the block that breaks down at
T+30 in a new month should be the one whose flow, carried forward by the demand shape, reaches its own fixed capacity.
Law (no block identity): u_s = f_s(T) * mu_s(slot T+6) / mu_s(slot T) / C_s with f_s(T) = block flow, mean of rows
T-2..T; mu_s = mean free-flow block flow by 5-min slot and day type; C_s = median flow in the block's congested cells.
mu, C and the blocks come from the OTHER month folds. P(block s in the T+30 queue) = sigmoid(a + b log u_s + c log u0_s)
with u0_s = f_s(T) / C_s, pooled over blocks and panels (fitted on the other folds' onset windows).
Compared on the month-CV onset windows (G42 windows, onset mornings, scored at T+30 as production):
  NN    : the no-memory blend (G42)            LAW : the law alone
  STACK : logit p' = a + b logit p_NN + c logit p_LAW (fitted on the other folds)
Audit 10/14: flows from rows T-3..T-1 (history [T-60, T)); penalty 1e-7; the hard/easy split is dropped.
usage: python g48_struct_onset.py"""
import sys, numpy as np, pandas as pd
from scipy.optimize import minimize
sys.path.insert(0, r"C:/Users/user/tfb/src"); sys.path.insert(0, r"C:/Users/user/tfb/twin")
from t2_eval_oof import T, oof_of
from t2_model import best_set
from cachelib import load, T2_PANELS
from onset_shape3_lib import raw_matrices, blocks

MEM = ["q25_nomem_s3", "q25_nomem_s4", "q31_nomem_wp05_s0", "q31_nomem_attn1_s0", "q31_nomem_r13_s0", "q31_nomem_drop1_s0"]
CUR = np.array([1, 1, 2, 2, 1, 1], float); CUR /= CUR.sum()
FOLD = {**{m: 0 for m in ("2030-06", "2030-07", "2030-08")}, **{m: 1 for m in ("2030-09", "2030-10", "2030-11")},
        **{m: 2 for m in ("2030-12", "2031-01", "2031-02")}}
O = [oof_of(m) for m in MEM]
keys = sorted(set.intersection(*[set(o) for o in O]) & {k for k in range(T.NWIN) if T.META["is_sel"][k]})
ons = [k for k in keys if str(T.META["cond"][k]) == "queue_onset" and (int(T.META["t"][k]) + 6) % 288 < 144]

W = {}
for p in T2_PANELS:
    kp = [k for k in ons if str(T.META["panel"][k]) == p]
    if not kp:
        continue
    P = load(p, "train"); V, Q, _ = raw_matrices(P); D = P.T // 288
    fold_row = np.array([FOLD.get(str(t)[:7], -1) for t in P.times])
    slot = np.arange(P.T) % 288; we = np.repeat(np.array([pd.Timestamp(str(x)).dayofweek >= 5 for x in P.day[:D]]), 288)[:P.T]
    cong = (V <= P.vcut[None, :]) & np.isfinite(V)
    for f in range(3):
        rows = fold_row != f; bid = blocks(V, P, rows); nb = bid.max() + 1
        fb = np.stack([np.nanmean(Q[:, bid == j], axis=1) for j in range(nb)], axis=1)            # (T, nb) block flow
        cb = np.stack([cong[:, bid == j].any(axis=1) for j in range(nb)], axis=1)
        C = np.array([np.nanmedian(fb[rows & cb[:, j], j]) if (rows & cb[:, j]).sum() >= 30 else np.nan for j in range(nb)])
        fbm = np.where(cb, np.nan, fb); mu = np.full((2, 288, nb), np.nan)
        for w_ in (0, 1):
            for s in range(288):
                sel = rows & (we == w_) & (slot == s)
                if sel.sum() > 3:
                    mu[w_, s] = np.nanmean(fbm[sel], axis=0)
        for k in kp:
            if FOLD[str(T.META["date"][k])[:7]] != f:
                continue
            t = int(T.META["t"][k]); s0 = slot[t]; w_ = int(we[t])
            ft = np.nanmean(fb[t - 3:t], axis=0)              # rows T-3..T-1 (history is [T-60, T))
            ratio = mu[w_, min(s0 + 6, 287)] / mu[w_, s0 - 2]
            u = ft * ratio / C; u0 = ft / C
            W[k] = (np.log(np.clip(np.nan_to_num(u, nan=0.05), 0.05, 3)), np.log(np.clip(np.nan_to_num(u0, nan=0.05), 0.05, 3)), np.isfinite(C), bid)
    print(p, "windows", sum(1 for k in kp if k in W), flush=True)
    load.cache_clear(); del V, Q, cong

rec = []
for k in ons:
    if k not in W:
        continue
    lab, el = T.LAB[k]; y = lab.astype(bool); e = el.astype(bool)
    pr = np.clip(np.tensordot(CUR, np.stack([o[k] for o in O]), 1), 1e-5, 1 - 1e-5)
    lu, lu0, cand, bid = W[k]; nb = bid.max() + 1
    yb = np.array([(y[5] & e[5] & (bid == j)).any() for j in range(nb)])
    rec.append(dict(k=k, f=FOLD[str(T.META["date"][k])[:7]], pnl=str(T.META["panel"][k]), cl=str(T.META["panel"][k]) + str(T.META["date"][k])[:10],
                    p=pr, y=y, e=e, lu=lu, lu0=lu0, cand=cand, bid=bid, yb=yb))
mode_blk = {}
for pnl in {r["pnl"] for r in rec}:
    cnt = {}
    for r in rec:
        if r["pnl"] == pnl:
            for j in np.flatnonzero(r["yb"]):
                cnt[j] = cnt.get(j, 0) + 1
    mode_blk[pnl] = max(cnt, key=cnt.get) if cnt else -1


def fit(A, y):
    def f(w):
        z = A @ w; return np.mean(np.logaddexp(0, z) - y * z) + 1e-7 * (w @ w), A.T @ (1 / (1 + np.exp(-z)) - y) / len(y) + 2e-7 * w
    return minimize(f, np.zeros(A.shape[1]), jac=True, method="L-BFGS-B").x


def law_design(rs):
    X, Y = [], []
    for r in rs:
        m = r["cand"]; X.append(np.column_stack([np.ones(m.sum()), r["lu"][m], r["lu0"][m]])); Y.append(r["yb"][m].astype(float))
    return np.vstack(X), np.concatenate(Y)


def law_link(r, w):
    pb = np.where(r["cand"], 1 / (1 + np.exp(-(w[0] + w[1] * r["lu"] + w[2] * r["lu0"]))), 1e-4)
    return np.clip(pb[r["bid"]], 1e-5, 1 - 1e-5)


def iou(q, r):
    qq = np.zeros_like(r["p"]); qq[5] = q
    s = best_set(qq.ravel(), np.ones(qq.size, bool)).reshape(qq.shape)
    u = ((s | r["y"]) & r["e"]).sum(); return 1.0 if u == 0 else (s & r["y"] & r["e"]).sum() / u


lg = lambda x: np.log(x / (1 - x))
out = []
for f in range(3):
    tr = [r for r in rec if r["f"] != f]; te = [r for r in rec if r["f"] == f]
    wl = fit(*law_design(tr))
    Xs, Ys = [], []
    for r in tr:
        m = r["e"][5]; Xs.append(np.column_stack([np.ones(m.sum()), lg(r["p"][5][m]), lg(law_link(r, wl))[m]])); Ys.append(r["y"][5][m].astype(float))
    ws = fit(np.vstack(Xs), np.concatenate(Ys))
    print(f"fold {f}: law a,b,c = {np.round(wl, 3).tolist()} | stack a,b_nn,c_law = {np.round(ws, 3).tolist()}", flush=True)
    for r in te:
        pl = law_link(r, wl); ps = 1 / (1 + np.exp(-(ws[0] + ws[1] * lg(r["p"][5]) + ws[2] * lg(pl))))
        hard = not bool(r["yb"][mode_blk[r["pnl"]]]) if mode_blk[r["pnl"]] < len(r["yb"]) else True
        out.append(dict(cl=r["cl"], pnl=r["pnl"], hard=hard, NN=iou(r["p"][5], r), LAW=iou(pl, r), STACK=iou(ps, r)))
D = pd.DataFrame(out); D.to_csv(r"C:/Users/user/tfb/work/g48_struct_onset.csv", index=False); rng = np.random.default_rng(0)
for nm, g in (("all", D),):
    u = g.cl.unique(); idx = {x: np.flatnonzero(g.cl.to_numpy() == x) for x in u}; msg = []
    for a in ("LAW", "STACK"):
        d = (g[a] - g["NN"]).to_numpy()
        se = np.std([d[np.concatenate([idx[x] for x in rng.choice(u, len(u))])].mean() for _ in range(500)])
        msg.append(f"{a}-NN {d.mean():+.4f} (SE {se:.4f})")
    print(f"{nm}: windows {len(g)}, NN {g.NN.mean():.4f}, LAW {g.LAW.mean():.4f}, STACK {g.STACK.mean():.4f} | " + " | ".join(msg), flush=True)
print(D.groupby("pnl")[["NN", "LAW", "STACK"]].mean().round(3).to_string())
