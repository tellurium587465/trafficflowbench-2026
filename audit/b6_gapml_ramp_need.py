"""B6 (10/24, strategy X item 1, need test before building): do the ramp flows, which stay visible inside the Task 2 blank
(80 % observed there as everywhere, while the mainline is hidden), explain what the blank-cell model v1 still gets wrong?
Out-of-sample residuals of gapml_*_cv (trained Jun-Dec) on the Jan-Feb eval/selector windows of the T2 panels, every
eligible observed blank cell (subsampled 0.25). Ramp features per gap row k and link j (rows t+k of the ramp layer):
  on / off flow of the link's own ramps minus their interpolation between the pre (t-2..t) and post (t+19..t+21) means,
  the upstream-cumulative net ramp deviation per lane (whole corridor and the last 5 links), the corridor's on-ramp
  total relative to its pre/post interpolation.
Incremental R^2 of the ramp features: LightGBM on the residual with base features (k, v_lin, q_lin, pre_v, post_v, w,
lanes) vs base + ramp, 5-fold CV grouped by window. Pre-declared reading (calibration: gapml v2 cut the blank RMSE by 5 %
for LB +0.0001): R^2 gain < 0.05 on both channels -> stop (ledger: expected LB < +0.0002); otherwise build it into gapml,
capped at half a day, one LB submission (agreed with the strategy session 10/24 02:30, before the result).
usage: python b6_gapml_ramp_need.py"""
import sys, time, numpy as np, lightgbm as lgb, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, r"C:/Users/user/tfb/src")
from cachelib import load, T2_PANELS
from common import WORK
import t2_demand
from t2_demand import profile
from t2_arrival import _dmin
from t1_gapml import view_train, gap_features, G, L0

t0 = time.time(); MD = WORK + "/t1/"
gm = {ch: lgb.Booster(model_file=MD + f"gapml_{ch}_cv.txt") for ch in ("speed", "flow")}
Z = np.load(WORK + "/t2/gapml_windows.npz", allow_pickle=True); m = Z["is_eval"] | Z["is_sel"]
rng = np.random.default_rng(6); ROWS = []
BASE = ["k", "v_lin", "q_lin", "pre_v", "post_v", "w", "lanes"]


def interp(A, t):
    """(G, m) deviation of the ramp series A (T, m) on rows t+1..t+G from the line between the pre and post means"""
    with np.errstate(all="ignore"):
        pre = np.nanmean(A[t - 2:t + 1], 0); post = np.nanmean(A[t + G + 1:t + G + 4], 0)
    w = (np.arange(1, G + 1) / (G + 1.0))[:, None]
    return A[t + 1:t + G + 1] - (pre[None, :] * (1 - w) + post[None, :] * w)


for p in T2_PANELS:
    P = load(p, "train"); x = np.concatenate([[0.0], np.cumsum(P.length[:-1])]); dm = _dmin(P, x)
    li = {l: i for i, l in enumerate(P.links)}; rj = np.array([li[str(l)] for l in P.ramp_link]); on = P.ramp_type == "OR"
    wins = sorted({(int(t), str(c)) for pp, t, c in zip(Z["panel"][m], Z["t"][m], Z["cond"][m]) if str(pp) == p and str(P.times[int(t)])[:7] in ("2031-01", "2031-02")})
    for t, c in wins:
        if t + G + 31 > P.T or t < 30:
            continue
        V, Q = view_train(P, t)
        X, names, bv, bq, kk, jj = gap_features(P, V, Q, t, c, profile(P, str(P.times[t])[:7]), dm, x)   # as t1_gapml_apply
        rr = t + kk; tv = P.V[rr, jj].astype(float); tq = (P.Q[rr, jj] / P.lanes[jj]).astype(float)
        ok = P.E[rr, jj] & np.isfinite(tv) & np.isfinite(tq) & np.isfinite(bv) & np.isfinite(bq) & (rng.random(len(rr)) < 0.25)
        if not ok.any():
            continue
        D = np.nan_to_num(interp(P.R.astype(float), t))                    # (G, n_ramps), unobserved -> 0
        don = np.zeros((G, P.n)); doff = np.zeros((G, P.n))
        np.add.at(don.T, rj[on], D[:, on].T); np.add.at(doff.T, rj[~on], D[:, ~on].T)
        net = np.cumsum(don - doff, axis=1) / P.lanes[None, :]
        net5 = net - np.c_[np.zeros((G, 5)), net[:, :-5]]
        with np.errstate(all="ignore"):
            tot = np.nansum(P.R[:, on], 1).astype(float); pre, post = tot[t - 2:t + 1].mean(), tot[t + G + 1:t + G + 4].mean()
            lev = tot[t + 1:t + G + 1] / np.maximum(pre * (1 - np.arange(1, G + 1) / (G + 1.0)) + post * np.arange(1, G + 1) / (G + 1.0), 1.0)
        k0 = kk - 1
        R = np.column_stack([don[k0, jj], doff[k0, jj], net[k0, jj], net5[k0, jj], lev[k0]])
        ev = tv - (bv + gm["speed"].predict(X, num_threads=8)); eq = tq - (bq + gm["flow"].predict(X, num_threads=8))
        Bf = X[:, [names.index(b) for b in BASE]]
        ROWS.append((Bf[ok], R[ok], ev[ok], eq[ok], np.full(ok.sum(), len(ROWS))))
    print(p, "windows", len(wins), "cells", sum(len(r[2]) for r in ROWS), f"{time.time() - t0:.0f}s", flush=True)
    load.cache_clear(); t2_demand._PROF.clear()

B = np.vstack([r[0] for r in ROWS]); R = np.vstack([r[1] for r in ROWS]); EV = np.concatenate([r[2] for r in ROWS]); EQ = np.concatenate([r[3] for r in ROWS])
GR = np.concatenate([r[4] for r in ROWS]); fold = np.random.default_rng(7).integers(0, 5, GR.max() + 1)[GR]
PRM = dict(objective="regression", learning_rate=0.05, num_leaves=31, min_data_in_leaf=200, feature_fraction=0.9, verbose=-1, num_threads=8, seed=3)
for ch, y in (("speed km/h", EV), ("flow vph/lane", EQ)):
    out = {}
    for nm, F in (("base", B), ("base+ramp", np.column_stack([B, R]))):
        pr = np.zeros(len(y))
        for f in range(5):
            tr, te = fold != f, fold == f
            pr[te] = lgb.train(PRM, lgb.Dataset(F[tr], y[tr]), 300).predict(F[te], num_threads=8)
        out[nm] = 1 - np.mean((y - pr) ** 2) / np.mean((y - y.mean()) ** 2)
    print(f"{ch}: residual RMSE {np.sqrt(np.mean(y ** 2)):.2f} (n {len(y)}, windows {GR.max() + 1}) | CV R^2 base {out['base']:+.4f}, base+ramp {out['base+ramp']:+.4f} "
          f"-> ramp gain {out['base+ramp'] - out['base']:+.4f} (RMSE x{np.sqrt(max(1 - (out['base+ramp'] - out['base']), 0)):.3f})", flush=True)
print(f"done {time.time() - t0:.0f}s")
