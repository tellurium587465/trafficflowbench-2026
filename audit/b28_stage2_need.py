"""B28 (11/01, ledger #112, stage 0, literature audit T1 + T3): do the residuals of the Task 1 models carry information a second stage could use?
Feb harness of audit/b21 (f3big + gapml_cv, February out of both trainings, density clip). The filled field = visible cells, else the
first-stage prediction. Residuals on the target cells: r_v = truth - prediction (km/h), r_q = the same per lane. Features per target cell:
the prediction itself; the filled field at t-2, t-1, t+1, t+2 and link -2, -1, +1, +2 (v / vf, q / lanes); 3 x 3 and 5 x 5 means of it;
conservation N_fwd / L, N_bwd / L minus the cell's density; predicted congested, Task 1 regime, length of the hidden run along time,
inside a blank. Cross-validated LightGBM on the residual (lr 0.05, 31 leaves, min_data 500, 300 rounds), five folds by day of the month.
R^2 = 1 - SSE after / SSE before on the held-out folds, by class: congested (prediction <= v_cut), blank, free (the rest).
Also the need of the conservation features: R^2 of dN_obs on the observed flow difference x dt on visible cells.
Line (declared first): R^2 >= 0.05 on either channel in congested or blank -> stage 1; otherwise close.
usage: py -B -X utf8 b28_stage2_need.py [panels...]"""
import sys
src = open(r"C:/Users/user/tfb/audit/b21_floor_bound.py", encoding="utf-8").read().split("RES = {}")[0]
ARGS = sys.argv[1:]; sys.dont_write_bytecode = True; exec(compile(src, "b21", "exec"), globals())
import pandas as pd
from scipy.ndimage import uniform_filter
OUT = WORK + "/t1/b28_rows.npz"; MAXC = 120000; rows = []; cons = []
FN = [f"v_t{d}" for d in (-2, -1, 1, 2)] + [f"v_l{d}" for d in (-2, -1, 1, 2)] + [f"q_t{d}" for d in (-2, -1, 1, 2)] + [f"q_l{d}" for d in (-2, -1, 1, 2)] + \
     ["v_hat", "q_hat", "v_m3", "q_m3", "v_m5", "q_m5", "n_fwd", "n_bwd", "cong", "regime", "run", "blank"]


def run_len(H):
    R = np.zeros(H.shape, np.int32)
    for t in range(1, H.shape[0]):
        R[t] = np.where(H[t], R[t - 1] + 1, 0)
    for t in range(H.shape[0] - 2, -1, -1):
        R[t] = np.where(H[t], np.maximum(R[t], R[t + 1]), 0)
    return R


for p in (ARGS or PANELS):
    P0 = load(p, "train"); P, a0 = feb(P0); pi = PANELS.index(p)
    x = np.concatenate([[0.0], np.cumsum(P.length[:-1])]); dm = _dmin(P, x); prof = profile(P0, "2031-02")
    wl = sorted({(int(t) - a0, str(c)) for pp, t, c in zip(Z["panel"][WM], Z["t"][WM], Z["cond"][WM]) if str(pp) == p and 30 <= int(t) - a0 < P.T - G - 31})[:10]
    Pg = copy.copy(P); Pg.blank = P.blank.copy(); Pg.target = P.target.copy(); gap = np.zeros((P.T, P.n), bool)
    for t, _ in wl:
        gap[t + 1:t + G + 1] = True; Pg.blank[t - 12:t] = False; Pg.target[t - 12:t] = False
    Pg.blank |= gap & np.isfinite(P.V)
    F, _, _, _ = grid_features3(Pg); bv, bq = base_of3(F); r, c = np.nonzero(Pg.target)
    X, _ = rows3(Pg, F, r, c); X = np.column_stack([np.full(len(r), pi, np.float32), X])
    Vp = np.full((P.T, P.n), np.nan); Qp = Vp.copy()
    Vp[r, c] = np.clip((bv[r, c] + mv.predict(X, num_threads=6)) * P.vf[c], 3.0, 1.15 * P.vf[c])
    Qp[r, c] = np.clip((bq[r, c] + mq.predict(X, num_threads=6)) * P.lanes[c], 60.0, None)
    for t, cnd in wl:
        V, Q = view_train(Pg, t); Xg, _, gbv, gbq, kk, jj = gap_features(Pg, V, Q, t, cnd, prof, dm, x)
        rr = t + kk; tg = Pg.target[rr, jj] & np.isfinite(gbv) & np.isfinite(gbq)
        Vp[rr[tg], jj[tg]] = np.clip(gbv[tg] + gm["speed"].predict(Xg[tg], num_threads=6), 3.0, 1.15 * P.vf[jj[tg]])
        Qp[rr[tg], jj[tg]] = np.clip((gbq[tg] + gm["flow"].predict(Xg[tg], num_threads=6)) * P.lanes[jj[tg]], 60.0, None)
    Vb = np.maximum(Vp, Qp / P.kj[None, :]); del X, F
    vis = np.isfinite(P.V) & np.isfinite(P.Q) & ~Pg.target & ~Pg.blank
    Fv = np.where(vis, P.V, Vb) / P.vf[None, :]; Fq = np.where(vis, P.Q, Qp) / P.lanes[None, :]; T_, n = Fv.shape
    W_ = np.isfinite(Fv) & np.isfinite(Fq); Fv0 = np.nan_to_num(Fv); Fq0 = np.nan_to_num(Fq)
    m3 = uniform_filter(W_.astype(float), 3, mode="nearest"); m5 = uniform_filter(W_.astype(float), 5, mode="nearest")
    vm3 = uniform_filter(Fv0, 3, mode="nearest") / np.maximum(m3, 1e-6); qm3 = uniform_filter(Fq0, 3, mode="nearest") / np.maximum(m3, 1e-6)
    vm5 = uniform_filter(Fv0, 5, mode="nearest") / np.maximum(m5, 1e-6); qm5 = uniform_filter(Fq0, 5, mode="nearest") / np.maximum(m5, 1e-6)
    Qt = Fq * P.lanes[None, :]; Vk = Fv * P.vf[None, :]; L = P.length[None, :]; Nn = Qt / np.maximum(Vk, 1.0) * L; dt = 5.0 / 60
    inf = np.zeros_like(Qt) * np.nan; inf[:, 1:] = Qt[:, :-1]; net = (inf - Qt) * dt
    Nf = np.full_like(Nn, np.nan); Nf[1:] = Nn[:-1] + net[:-1]; Nb = np.full_like(Nn, np.nan); Nb[:-1] = Nn[1:] - net[:-1]
    kh = Qp / np.maximum(Vb, 1.0); run = run_len(~vis); idx = np.flatnonzero(np.isfinite(Vb[r, c]) & np.isfinite(Qp[r, c]) & np.isfinite(P.V[r, c]) & np.isfinite(P.Q[r, c]))
    rng = np.random.default_rng(pi); ing = gap[r[idx], c[idx]]; rest = idx[~ing]                # every cell of the simulated Task 2 blanks stays
    idx = np.r_[idx[ing], rng.choice(rest, min(MAXC, len(rest)), replace=False)]; r_, c_ = r[idx], c[idx]
    shift = lambda A, dt_, dl: A[np.clip(r_ + dt_, 0, T_ - 1), np.clip(c_ + dl, 0, n - 1)]
    cols = [shift(Fv, d, 0) for d in (-2, -1, 1, 2)] + [shift(Fv, 0, d) for d in (-2, -1, 1, 2)] + [shift(Fq, d, 0) for d in (-2, -1, 1, 2)] + [shift(Fq, 0, d) for d in (-2, -1, 1, 2)]
    cols += [Vb[r_, c_] / P.vf[c_], Qp[r_, c_] / P.lanes[c_], vm3[r_, c_], qm3[r_, c_], vm5[r_, c_], qm5[r_, c_], (Nf / L)[r_, c_] - kh[r_, c_], (Nb / L)[r_, c_] - kh[r_, c_],
             (Vb[r_, c_] <= P.vcut[c_]).astype(float), P.regime[r_, c_].astype(float) if P.regime.ndim == 2 else P.regime[r_].astype(float), run[r_, c_].astype(float), gap[r_, c_].astype(float)]
    rows.append(pd.DataFrame(dict(zip(FN, cols))).assign(panel=pi, day=r_ // 288, rv=P.V[r_, c_] - Vb[r_, c_], rq=(P.Q[r_, c_] - Qp[r_, c_]) / P.lanes[c_]).astype(np.float32))
    okc = vis[:-1] & vis[1:]; ok2 = okc[:, 1:] & vis[:-1, :-1]; Ko = P.Q / np.maximum(P.V, 1.0) * L; dN = (Ko[1:] - Ko[:-1])[:, 1:][ok2]; flx = ((P.Q[:-1, :-1] - P.Q[:-1, 1:]) * dt)[ok2]
    b = np.polyfit(flx, dN, 1); cons.append(dict(panel=p, n=int(ok2.sum()), r2=1 - np.var(dN - np.polyval(b, flx)) / np.var(dN), slope=b[0]))
    print(p, len(idx), "cells | congested", round(float(rows[-1].cong.mean()), 4), "blank", round(float(rows[-1].blank.mean()), 4), flush=True); load.cache_clear()
D = pd.concat(rows, ignore_index=True); D.to_parquet(WORK + "/t1/b28_rows.parquet", index=False)
print("cells", len(D), "| R2 of dN on the flow difference:", pd.DataFrame(cons).round(4).to_string(index=False), flush=True)
PRM = dict(objective="regression", learning_rate=0.05, num_leaves=31, min_data_in_leaf=500, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=2.0, num_threads=6, verbose=-1, seed=7)
fold = (D.day.to_numpy() % 5).astype(int); cl = {"congested": D.cong.to_numpy() > 0.5, "blank": D.blank.to_numpy() > 0.5}; cl["free"] = ~(cl["congested"] | cl["blank"])
for ch in ("rv", "rq"):
    y = D[ch].to_numpy(); pr = np.zeros(len(y))
    for f in range(5):
        m = lgb.train(PRM, lgb.Dataset(D.loc[fold != f, FN], y[fold != f]), 300); pr[fold == f] = m.predict(D.loc[fold == f, FN], num_threads=6)
    print(ch, {k: dict(n=int(v.sum()), rmse=round(float(np.sqrt((y[v] ** 2).mean())), 4), r2=round(float(1 - ((y[v] - pr[v]) ** 2).sum() / (y[v] ** 2).sum()), 4)) for k, v in cl.items()}, "| all", round(float(1 - ((y - pr) ** 2).sum() / (y ** 2).sum()), 4), flush=True)
