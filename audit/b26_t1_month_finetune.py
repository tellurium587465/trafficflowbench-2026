"""B26 (10/29, ledger #64, after #62 passed): supervised fine-tuning of the T1 models inside the month, one setting, on the
Feb harness of B25 / B21 (f3big + gapml_cv, both trained without February; 10 simulated T2 blanks per panel; density clip).
The month's VISIBLE cells are the only labels; the harness's hidden cells (T1 targets and the 10 blanks) are never used.
  f3big : artificial hide = the month's own T1 hide pattern (P.blank) shifted by 1, 2, 3 days, on the visible cells (hidden
          cells stay hidden); features rebuilt on that view; continued boosting from f3big (new trees only: lr 0.05, 100
          rounds, 63 leaves), the hard-cell weighting of t1_full (min_vr_nb < 0.9 x10).
  gapml : the month's pseudo-windows (gapml_windows.npz, February, blanks not overlapping the 10 scored blanks, <= 150 per
          panel) on a view that keeps every hidden cell hidden (view_train un-hides the history rows from P.V = the truth,
          a leak here), labels on visible cells only, b1's subsampling; continued boosting from gapml_cv (same setting).
Scored on the harness's target cells with the untouched view: base / f3big only / gapml only / both;
0.35 dS_state + 0.15 dS_phys per panel, mean and SE. Line (#64): both >= +0.001 and no significant harm.
usage: py b26_t1_month_finetune.py [panels...]"""
import sys, copy, time, numpy as np, lightgbm as lgb, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, r"C:/Users/user/tfb/src")
from cachelib import load
from common import PANELS, WORK
from t1_features3 import grid_features3, rows3, base_of3
from t2_demand import profile
from t2_arrival import _dmin
from t1_gapml import view_train, gap_features, G, L0, PARAMS, _local
from metrics import state_score, physics_score

MD = WORK + "/t1/"
mv = lgb.Booster(model_file=MD + "speed_f3big.txt"); mq = lgb.Booster(model_file=MD + "flow_f3big.txt")
gm = {ch: lgb.Booster(model_file=MD + f"gapml_{ch}_cv.txt") for ch in ("speed", "flow")}
Z = np.load(WORK + "/t2/gapml_windows.npz", allow_pickle=True); WM = Z["is_sel"]
FT_F = dict(objective="regression", learning_rate=0.05, num_leaves=63, min_data_in_leaf=100, feature_fraction=0.8, bagging_fraction=0.8,
            bagging_freq=1, lambda_l2=1.0, num_threads=16, verbose=-1, seed=26, force_row_wise=True, deterministic=True)
FT_G = dict(PARAMS, num_leaves=63, num_threads=16, seed=26)
NR, NROW = 100, 200000
rng = np.random.default_rng(26); t0 = time.time()


def feb(P):
    f = np.flatnonzero([s.startswith("2031-02") for s in P.times]); a, b = f[0], f[-1] + 1; Q = copy.copy(P)
    for k in ("times", "V", "Q", "O", "P", "E", "regime", "target", "blank", "R", "RE"):
        setattr(Q, k, getattr(P, k)[a:b].copy())
    Q.T = b - a; Q.day = np.array([t[:10] for t in Q.times[::288]])
    return Q, a


def view_ft(P, t):
    """view_train without un-hiding the history rows: every cell hidden in P (targets, blanks) stays hidden."""
    V = _local(P.V, t, P.T); Q = _local(P.Q, t, P.T); bl = _local(P.blank, t, P.T, fill=0).astype(bool)
    V[bl] = np.nan; Q[bl] = np.nan; V[L0 + 1:L0 + 1 + G] = np.nan; Q[L0 + 1:L0 + 1 + G] = np.nan
    ok = np.isfinite(V) & np.isfinite(Q); V[~ok] = np.nan; Q[~ok] = np.nan
    return V, Q


def score(Pg, V, Q, val):
    return state_score(Pg, V, Q), physics_score(Pg, V, Q, validity=val)[2]


RES = {}
for p in (sys.argv[1:] or PANELS):
    P0 = load(p, "train"); P, a0 = feb(P0); pi = PANELS.index(p)
    x = np.concatenate([[0.0], np.cumsum(P.length[:-1])]); dm = _dmin(P, x); prof = profile(P0, "2031-02")
    wl = sorted({(int(t) - a0, str(c)) for pp, t, c in zip(Z["panel"][WM], Z["t"][WM], Z["cond"][WM])
                 if str(pp) == p and 30 <= int(t) - a0 < P.T - G - 31})[:10]
    Pg = copy.copy(P); Pg.blank = P.blank.copy(); Pg.target = P.target.copy(); gap = np.zeros((P.T, P.n), bool)
    for t, _ in wl:
        gap[t + 1:t + G + 1] = True; Pg.blank[t - 12:t] = False; Pg.target[t - 12:t] = False
    Pg.blank |= gap & np.isfinite(P.V)
    vis = np.isfinite(P.V) & np.isfinite(P.Q) & P.E & ~Pg.blank & ~Pg.target           # the only cells whose truth is used
    # ---- f3big fine-tuning on artificial hides of visible cells
    Xs, yv, yq, ws = [], [], [], []
    for k in (1, 2, 3):
        art = np.roll(P.blank, 288 * k, axis=0) & vis
        Pa = copy.copy(Pg); Pa.blank = Pg.blank | art; Pa.target = art
        Fa, _, _, _ = grid_features3(Pa); bva, bqa = base_of3(Fa); r, c = np.nonzero(art)
        hard = np.nan_to_num(Fa["min_vr_nb"][r, c], nan=1.0) < 0.9; pr = np.where(hard, 10.0, 1.0); pr = pr / pr.sum()
        idx = rng.choice(len(r), size=min(NROW, len(r)), replace=False, p=pr); r, c = r[idx], c[idx]
        X, _ = rows3(Pa, Fa, r, c); Xs.append(np.column_stack([np.full(len(r), pi, np.float32), X]))
        yv.append(P.V[r, c] / P.vf[c] - bva[r, c]); yq.append(P.Q[r, c] / P.lanes[c] - bqa[r, c]); ws.append(1.0 / (pr[idx] * len(pr)))
        del Fa
    Xf = np.vstack(Xs); wf = np.concatenate(ws); ok = np.isfinite(np.concatenate(yv)) & np.isfinite(np.concatenate(yq))
    mvf = lgb.train(FT_F, lgb.Dataset(Xf[ok], np.concatenate(yv)[ok], weight=wf[ok], categorical_feature=[0]), NR, init_model=mv)
    mqf = lgb.train(FT_F, lgb.Dataset(Xf[ok], np.concatenate(yq)[ok], weight=wf[ok], categorical_feature=[0]), NR, init_model=mq)
    nf = int(ok.sum()); del Xf, Xs
    # ---- gapml fine-tuning on the month's pseudo-windows (visible labels only, no un-hiding)
    used = {t for t, _ in wl}; cand = [(int(t) - a0, str(c)) for pp, t, c in zip(Z["panel"], Z["t"], Z["cond"])
                                       if str(pp) == p and 30 <= int(t) - a0 < P.T - G - 31]
    cand = sorted({w for w in cand if all(abs(w[0] - u) > G + 1 for u in used)})
    cand = [cand[i] for i in sorted(rng.choice(len(cand), size=min(150, len(cand)), replace=False))] if cand else []
    GX, gyv, gyq, gw = [], [], [], []
    for t, cnd in cand:
        V, Q = view_ft(Pg, t); X, names, gbv, gbq, kk, jj = gap_features(Pg, V, Q, t, cnd, prof, dm, x)
        rr = t + kk; ok_ = (rr < P.T) & np.isfinite(gbv) & np.isfinite(gbq); rr_ = np.minimum(rr, P.T - 1)
        ok_ &= vis[rr_, jj]
        f = {nm: X[:, i] for i, nm in enumerate(names)}
        near = (np.minimum.reduce([f["dup_pre"], f["ddn_pre"], f["dup_post"], f["ddn_post"]]) <= 3.0) | (np.nan_to_num(f["v_lin"], nan=1) < 0.85)
        pk = np.where(near, 0.5, 0.08); keep = ok_ & (rng.random(len(ok_)) < pk)
        GX.append(X[keep]); gyv.append(P.V[rr_[keep], jj[keep]] - gbv[keep]); gyq.append(P.Q[rr_[keep], jj[keep]] / P.lanes[jj[keep]] - gbq[keep]); gw.append(1.0 / pk[keep])
    gmf = None; ng = 0
    if GX and sum(len(a) for a in GX) > 1000:
        GXa = np.vstack(GX); gwa = np.concatenate(gw); ng = len(GXa)
        gmf = {ch: lgb.train(FT_G, lgb.Dataset(GXa, np.concatenate(y), weight=gwa), NR, init_model=gm[ch]) for ch, y in (("speed", gyv), ("flow", gyq))}
        del GXa
    # ---- predictions on the harness's scored cells with the untouched view
    F, _, _, _ = grid_features3(Pg); bv, bq = base_of3(F); r, c = np.nonzero(Pg.target)
    X, _ = rows3(Pg, F, r, c); X = np.column_stack([np.full(len(r), pi, np.float32), X])
    out = {}
    for nm, (a_v, a_q, g_) in {"base": (mv, mq, gm), "f3big_ft": (mvf, mqf, gm), "gapml_ft": (mv, mq, gmf or gm), "both": (mvf, mqf, gmf or gm)}.items():
        Vp = np.full((P.T, P.n), np.nan); Qp = Vp.copy()
        Vp[r, c] = np.clip((bv[r, c] + a_v.predict(X, num_threads=16)) * P.vf[c], 3.0, 1.15 * P.vf[c])
        Qp[r, c] = np.clip((bq[r, c] + a_q.predict(X, num_threads=16)) * P.lanes[c], 60.0, None)
        for t, cnd in wl:
            V, Q = view_train(Pg, t); Xg, _, gbv, gbq, kk, jj = gap_features(Pg, V, Q, t, cnd, prof, dm, x)
            rr = t + kk; tg = Pg.target[rr, jj] & np.isfinite(gbv) & np.isfinite(gbq)
            Vp[rr[tg], jj[tg]] = np.clip(gbv[tg] + g_["speed"].predict(Xg[tg], num_threads=16), 3.0, 1.15 * P.vf[jj[tg]])
            Qp[rr[tg], jj[tg]] = np.clip((gbq[tg] + g_["flow"].predict(Xg[tg], num_threads=16)) * P.lanes[jj[tg]], 60.0, None)
        Vp = np.maximum(Vp, Qp / P.kj[None, :]); out[nm] = score(Pg, Vp, Qp, P.ramp_validity())
    d = {nm: 0.35 * (v[0] - out["base"][0]) + 0.15 * (v[1] - out["base"][1]) for nm, v in out.items() if nm != "base"}
    RES[p] = d
    print(p, f"f3 rows {nf} | gap windows {len(cand)} rows {ng} | " + " | ".join(f"{k} {v:+.4f}" for k, v in d.items()), f"{time.time() - t0:.0f}s", flush=True)
    load.cache_clear()

print("\n0.35 dS_state + 0.15 dS_phys vs the base, mean over panels (SE):")
for k in ("f3big_ft", "gapml_ft", "both"):
    v = np.array([RES[p][k] for p in RES]); print(f"  {k:9s} {v.mean():+.4f} (SE {v.std(ddof=1) / np.sqrt(len(v)):.4f}) | panels > 0: {(v > 0).sum()}/{len(v)}")
v = np.array([RES[p]["both"] for p in RES]); m, se = v.mean(), v.std(ddof=1) / np.sqrt(len(v))
print("line (#64):", "build for March (>= +0.001, no harm)" if m >= 0.001 and m / se > -2 else ("closed: point positive, below the line" if m > 0 else "closed"))
