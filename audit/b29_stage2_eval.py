"""B29 (11/01, ledger #112, stage 1): the second-stage residual models on the Feb harness, the declared test. Features and first stage as B28 on all
target cells. Speed and flow LightGBM on the residuals (lr 0.03, 31 leaves, min_data 500, 450 rounds, fixed before the result), trained on B28's
sampled rows. Output v + r (lambda = 1), then the production chain: ceiling, density clip, FD 0.75, boundary model (bnd3 w 0.75) on both arms.
  halves : train on the odd days of February, apply to the even days, and the reverse
  panels : train on the other nine panels, apply to the panel
Reads 0.35 dS_state + 0.15 dS_phys per panel. Line: both halves >= 0 and at least 7 of 10 panels positive -> LB pair; otherwise closed.
usage: py -B -X utf8 b29_stage2_eval.py"""
import os, sys, pickle
src = open(r"C:/Users/user/tfb/audit/b21_floor_bound.py", encoding="utf-8").read().split("RES = {}")[0]
sys.argv = sys.argv[:1]; sys.dont_write_bytecode = True; exec(compile(src, "b21", "exec"), globals())
import pandas as pd
from scipy.ndimage import uniform_filter
from t1_bnd import released, selector, build
b28 = open(r"C:/Users/user/tfb/audit/b28_stage2_need.py", encoding="utf-8").read(); exec(b28[b28.index('FN = ['):b28.index('for p in (ARGS or PANELS):')], globals())
CD = WORK + "/t1/b29/"; os.makedirs(CD, exist_ok=True)
BM = {ch: lgb.Booster(model_file=MD + f"bnd3_{ch}_final.txt") for ch in ("speed", "flow")}


def stage1(p):
    fn = CD + f"{p}.pkl"
    if os.path.exists(fn):
        return pickle.load(open(fn, "rb"))
    P0 = load(p, "train"); P, a0 = feb(P0); pi = PANELS.index(p); x = np.concatenate([[0.0], np.cumsum(P.length[:-1])]); dm = _dmin(P, x); prof = profile(P0, "2031-02")
    wl = sorted({(int(t) - a0, str(c)) for pp, t, c in zip(Z["panel"][WM], Z["t"][WM], Z["cond"][WM]) if str(pp) == p and 30 <= int(t) - a0 < P.T - G - 31})[:10]
    Pg = copy.copy(P); Pg.blank = P.blank.copy(); Pg.target = P.target.copy(); gap = np.zeros((P.T, P.n), bool)
    for t, _ in wl:
        gap[t + 1:t + G + 1] = True; Pg.blank[t - 12:t] = False; Pg.target[t - 12:t] = False
    Pg.blank |= gap & np.isfinite(P.V)
    F, _, _, _ = grid_features3(Pg); bv, bq = base_of3(F); r, c = np.nonzero(Pg.target)
    X, _ = rows3(Pg, F, r, c); X = np.column_stack([np.full(len(r), pi, np.float32), X])
    Vp = np.full((P.T, P.n), np.nan); Qp = Vp.copy()
    Vp[r, c] = np.clip((bv[r, c] + mv.predict(X, num_threads=6)) * P.vf[c], 3.0, 1.15 * P.vf[c]); Qp[r, c] = np.clip((bq[r, c] + mq.predict(X, num_threads=6)) * P.lanes[c], 60.0, None)
    for t, cnd in wl:
        V, Q = view_train(Pg, t); Xg, _, gbv, gbq, kk, jj = gap_features(Pg, V, Q, t, cnd, prof, dm, x)
        rr = t + kk; tg = Pg.target[rr, jj] & np.isfinite(gbv) & np.isfinite(gbq)
        Vp[rr[tg], jj[tg]] = np.clip(gbv[tg] + gm["speed"].predict(Xg[tg], num_threads=6), 3.0, 1.15 * P.vf[jj[tg]]); Qp[rr[tg], jj[tg]] = np.clip((gbq[tg] + gm["flow"].predict(Xg[tg], num_threads=6)) * P.lanes[jj[tg]], 60.0, None)
    Vb = np.maximum(Vp, Qp / P.kj[None, :]); del X
    Vr, Qr = released(Pg, Pg.V, Pg.Q); sel, cc, ff = selector(Pg, Vr); rb, cb = np.nonzero(Pg.target & ~gap & sel); Xb, _ = build(Pg, F, Vr, Qr, rb, cb, cc, ff, rows3)
    vbd = np.clip((bv[rb, cb] + BM["speed"].predict(Xb, num_threads=6)) * P.vf[cb], 3.0, 1.15 * P.vf[cb]); qbd = np.clip((bq[rb, cb] + BM["flow"].predict(Xb, num_threads=6)) * P.lanes[cb], 60.0, None); del F, Xb
    vis = np.isfinite(P.V) & np.isfinite(P.Q) & ~Pg.target & ~Pg.blank
    Fv = np.where(vis, P.V, Vb) / P.vf[None, :]; Fq = np.where(vis, P.Q, Qp) / P.lanes[None, :]; T_, n = Fv.shape
    W_ = np.isfinite(Fv) & np.isfinite(Fq); Fv0 = np.nan_to_num(Fv); Fq0 = np.nan_to_num(Fq); m3 = uniform_filter(W_.astype(float), 3, mode="nearest"); m5 = uniform_filter(W_.astype(float), 5, mode="nearest")
    vm3 = uniform_filter(Fv0, 3, mode="nearest") / np.maximum(m3, 1e-6); qm3 = uniform_filter(Fq0, 3, mode="nearest") / np.maximum(m3, 1e-6); vm5 = uniform_filter(Fv0, 5, mode="nearest") / np.maximum(m5, 1e-6); qm5 = uniform_filter(Fq0, 5, mode="nearest") / np.maximum(m5, 1e-6)
    Qt = Fq * P.lanes[None, :]; Vk = Fv * P.vf[None, :]; L = P.length[None, :]; Nn = Qt / np.maximum(Vk, 1.0) * L; dt = 5.0 / 60
    inf = np.zeros_like(Qt) * np.nan; inf[:, 1:] = Qt[:, :-1]; net = (inf - Qt) * dt; Nf = np.full_like(Nn, np.nan); Nf[1:] = Nn[:-1] + net[:-1]; Nb = np.full_like(Nn, np.nan); Nb[:-1] = Nn[1:] - net[:-1]
    kh = Qp / np.maximum(Vb, 1.0); run = run_len(~vis); ok = np.isfinite(Vb[r, c]) & np.isfinite(Qp[r, c]); r_, c_ = r[ok], c[ok]
    shift = lambda A, dt_, dl: A[np.clip(r_ + dt_, 0, T_ - 1), np.clip(c_ + dl, 0, n - 1)]
    cols = [shift(Fv, d, 0) for d in (-2, -1, 1, 2)] + [shift(Fv, 0, d) for d in (-2, -1, 1, 2)] + [shift(Fq, d, 0) for d in (-2, -1, 1, 2)] + [shift(Fq, 0, d) for d in (-2, -1, 1, 2)]
    cols += [Vb[r_, c_] / P.vf[c_], Qp[r_, c_] / P.lanes[c_], vm3[r_, c_], qm3[r_, c_], vm5[r_, c_], qm5[r_, c_], (Nf / L)[r_, c_] - kh[r_, c_], (Nb / L)[r_, c_] - kh[r_, c_],
             (Vb[r_, c_] <= P.vcut[c_]).astype(float), P.regime[r_].astype(float) if P.regime.ndim == 1 else P.regime[r_, c_].astype(float), run[r_, c_].astype(float), gap[r_, c_].astype(float)]
    out = dict(X=np.column_stack(cols).astype(np.float32), r=r_, c=c_, Vb=Vb, Qp=Qp, rb=rb, cb=cb, vbd=vbd, qbd=qbd, gap=gap, wl=wl, cap=np.nanmax(np.where(np.isfinite(P0.V[:a0]), P0.V[:a0], np.nan), axis=0))
    pickle.dump(out, open(fn, "wb"), protocol=4); print(p, "first stage", len(r_), "cells", flush=True); load.cache_clear(); return out


def harness(p, S):
    P0 = load(p, "train"); P, a0 = feb(P0); Pg = copy.copy(P); Pg.blank = P.blank.copy(); Pg.target = P.target.copy()
    for t, _ in S["wl"]:
        Pg.blank[t - 12:t] = False; Pg.target[t - 12:t] = False
    Pg.blank |= S["gap"] & np.isfinite(P.V); return P, Pg


def post(P, S, V, Q):
    v = np.maximum(np.minimum(V, S["cap"][None, :]), Q / P.kj[None, :]); cong = v < 0.6 * P.vf[None, :]; k = np.maximum(P.kj[None, :] - Q / P.w[None, :], 1e-3); v = np.where(cong, 0.25 * v + 0.75 * Q / k, v); Q2 = Q.copy()
    rb, cb = S["rb"], S["cb"]; vv = 0.75 * S["vbd"] + 0.25 * v[rb, cb]; qq = 0.75 * S["qbd"] + 0.25 * Q[rb, cb]; vv = np.maximum(np.minimum(vv, S["cap"][cb]), qq / P.kj[cb])
    co = vv < 0.6 * P.vf[cb]; kk = np.maximum(P.kj[cb] - qq / P.w[cb], 1e-3); v[rb, cb] = np.where(co, 0.25 * vv + 0.75 * qq / kk, vv); Q2[rb, cb] = qq
    return np.maximum(v, Q2 / P.kj[None, :]), Q2


D = pd.read_parquet(WORK + "/t1/b28_rows.parquet"); PRM = dict(objective="regression", learning_rate=0.03, num_leaves=31, min_data_in_leaf=500, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=2.0, num_threads=6, verbose=-1, seed=7)
ST = {p: stage1(p) for p in PANELS}; print("first stage ready", flush=True)


def fit(mask):
    return {ch: lgb.train(PRM, lgb.Dataset(D.loc[mask, FN].to_numpy(), D.loc[mask, ch].to_numpy()), 450) for ch in ("rv", "rq")}


odd = (D.day.to_numpy() % 2).astype(bool); Mh = {1: fit(odd), 0: fit(~odd)}; print("halves fitted", flush=True); RES = []
for p in PANELS:
    S = ST[p]; P, Pg = harness(p, S); pi = PANELS.index(p); val = P.ramp_validity(); day = S["r"] // 288; Mp = fit(D.panel.to_numpy() != pi); arms = {}
    rvh = np.zeros(len(day)); rqh = rvh.copy()
    for par in (0, 1):
        m = (day % 2) == par; mdl = Mh[1 - par]; rvh[m] = mdl["rv"].predict(S["X"][m], num_threads=6); rqh[m] = mdl["rq"].predict(S["X"][m], num_threads=6)
    rvp = Mp["rv"].predict(S["X"], num_threads=6); rqp = Mp["rq"].predict(S["X"], num_threads=6)

    def apply(rv, rq):
        V = S["Vb"].copy(); Q = S["Qp"].copy(); V[S["r"], S["c"]] = np.clip(V[S["r"], S["c"]] + rv, 3.0, 1.15 * P.vf[S["c"]]); Q[S["r"], S["c"]] = np.clip(Q[S["r"], S["c"]] + rq * P.lanes[S["c"]], 60.0, None); return post(P, S, V, Q)

    def sc(Pm, V, Q):
        s = state_score(Pm, V, Q); _, _, ph = physics_score(Pm, V, Q, validity=val); return s, ph

    Vb0, Qb0 = post(P, S, S["Vb"], S["Qp"]); Vh, Qh = apply(rvh, rqh); Vp_, Qp_ = apply(rvp, rqp); r = dict(panel=p)
    a = sc(Pg, Vb0, Qb0); b = sc(Pg, Vh, Qh); c_ = sc(Pg, Vp_, Qp_); r.update(base_state=a[0], base_phys=a[1], ds=b[0] - a[0], dp=b[1] - a[1], obj=0.35 * (b[0] - a[0]) + 0.15 * (b[1] - a[1]), obj_panel=0.35 * (c_[0] - a[0]) + 0.15 * (c_[1] - a[1]))
    for par in (0, 1):
        Pm = copy.copy(Pg); rows_ = ((np.arange(P.T) // 288) % 2) == par; Pm.target = Pg.target & rows_[:, None]; a2 = sc(Pm, Vb0, Qb0); b2 = sc(Pm, Vh, Qh); r[f"obj_half{par}"] = 0.35 * (b2[0] - a2[0]) + 0.15 * (b2[1] - a2[1])
    RES.append(r); print({k: (round(v, 5) if isinstance(v, float) else v) for k, v in r.items()}, flush=True); load.cache_clear()
R = pd.DataFrame(RES); R.to_csv(WORK + "/t1/b29_stage2.csv", index=False); print(R.drop(columns="panel").mean().round(5).to_string())
print("positive panels (halves cross-fit)", int((R.obj > 0).sum()), "of", len(R), "| panel-out", int((R.obj_panel > 0).sum()), "| half 0 mean", round(R.obj_half0.mean(), 5), "| half 1 mean", round(R.obj_half1.mean(), 5))
