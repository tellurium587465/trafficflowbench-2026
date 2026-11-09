"""B14 (10/29, strategy lane 12 (b), orientation before the gapml feature): what does a low-rank completion of the month
add inside the simulated Task 2 blanks? Larmuseau (10/27) reports +0.003 on the board from "a low-rank tensor completion of
the month itself" on top of a same-slot profile averager; our gapml sees no same-month other-day information (its demand
profile is the training months' weekday x time of day).
Feb harness of audit/b8 / b11 (f3big on the Task 1 targets + gapml_cv inside 10 simulated blanks per panel + density clip).
Completion: per panel and channel (v / vf, q / cap), the Feb matrix days x (288 * links) with the participant's view
observed (finite and not blank), hard-impute: start from the same-slot mean over the other days, then rank-r SVD
projections (40 iterations). Inside the blanks only: v = (1 - w) gapml + w completion (same for q), density clip after.
Columns: same-slot mean (r = 0) and r in (2, 4, 8), w in (0.25, 0.5, 1). Prints per-variant mean dS_state, dS_phys and
0.35 dS_state + 0.15 dS_phys, and the speed RMSE on the scored blank cells.
usage: py b14_lrtc_feb.py [panels...]"""
import sys, copy, numpy as np, lightgbm as lgb, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, r"C:/Users/user/tfb/src")
from cachelib import load
from common import PANELS, WORK
from t1_features3 import grid_features3, rows3, base_of3
from t2_demand import profile
from t2_arrival import _dmin
from t1_gapml import view_train, gap_features, G
from metrics import state_score, physics_score

MD = WORK + "/t1/"
mv = lgb.Booster(model_file=MD + "speed_f3big.txt"); mq = lgb.Booster(model_file=MD + "flow_f3big.txt")
gm = {ch: lgb.Booster(model_file=MD + f"gapml_{ch}_cv.txt") for ch in ("speed", "flow")}
Z = np.load(WORK + "/t2/gapml_windows.npz", allow_pickle=True); WM = Z["is_sel"]
RANKS, WS = (0, 2, 4, 8), (0.25, 0.5, 1.0)


def feb(P):
    f = np.flatnonzero([s.startswith("2031-02") for s in P.times]); a, b = f[0], f[-1] + 1; Q = copy.copy(P)
    for k in ("times", "V", "Q", "O", "P", "E", "regime", "target", "blank", "R", "RE"):
        setattr(Q, k, getattr(P, k)[a:b].copy())
    Q.T = b - a; Q.day = np.array([t[:10] for t in Q.times[::288]])
    return Q, a


def complete(X, obs, ranks, iters=40):
    """X (T, n) with T = days * 288; obs bool. Returns {r: completed (T, n)}; r = 0 is the same-slot mean over days."""
    D = X.shape[0] // 288; M = X[:D * 288].reshape(D, -1); O = obs[:D * 288].reshape(D, -1)
    with np.errstate(all="ignore"):
        mu = np.nanmean(np.where(O, M, np.nan), axis=0)
    mu = np.where(np.isfinite(mu), mu, np.nanmean(mu)); out = {}
    for r in ranks:
        Y = np.where(O, M, mu[None, :])
        if r > 0:
            for _ in range(iters):
                U, s, Vt = np.linalg.svd(Y - mu[None, :], full_matrices=False)
                L = mu[None, :] + (U[:, :r] * s[:r]) @ Vt[:r]; Y = np.where(O, M, L)
            Y = L
        else:
            Y = np.broadcast_to(mu, M.shape)
        full = np.full_like(X, np.nan); full[:D * 288] = Y.reshape(D * 288, -1); out[r] = full
    return out


ACC = {}; RM = {}; npn = 0; CELLS = []
for p in (sys.argv[1:] or PANELS):
    P0 = load(p, "train"); P, a0 = feb(P0); pi = PANELS.index(p)
    x = np.concatenate([[0.0], np.cumsum(P.length[:-1])]); dm = _dmin(P, x); prof = profile(P0, "2031-02")
    wl = sorted({(int(t) - a0, str(c)) for pp, t, c in zip(Z["panel"][WM], Z["t"][WM], Z["cond"][WM])
                 if str(pp) == p and 30 <= int(t) - a0 < P.T - G - 31})[:10]
    Pg = copy.copy(P); Pg.blank = P.blank.copy(); Pg.target = P.target.copy(); gap = np.zeros((P.T, P.n), bool)
    for t, _ in wl:
        gap[t + 1:t + G + 1] = True; Pg.blank[t - 12:t] = False; Pg.target[t - 12:t] = False
    Pg.blank |= gap & np.isfinite(P.V)
    F, _, _, _ = grid_features3(Pg); bv, bq = base_of3(F); r, c = np.nonzero(Pg.target)
    X, _ = rows3(Pg, F, r, c); X = np.column_stack([np.full(len(r), pi, np.float32), X])
    Vp = np.full((P.T, P.n), np.nan); Qp = Vp.copy()
    Vp[r, c] = np.clip((bv[r, c] + mv.predict(X, num_threads=8)) * P.vf[c], 3.0, 1.15 * P.vf[c])
    Qp[r, c] = np.clip((bq[r, c] + mq.predict(X, num_threads=8)) * P.lanes[c], 60.0, None)
    for t, cnd in wl:
        V, Q = view_train(Pg, t); Xg, _, gbv, gbq, kk, jj = gap_features(Pg, V, Q, t, cnd, prof, dm, x)
        rr = t + kk; tg = Pg.target[rr, jj] & np.isfinite(gbv) & np.isfinite(gbq)
        Vp[rr[tg], jj[tg]] = np.clip(gbv[tg] + gm["speed"].predict(Xg[tg], num_threads=8), 3.0, 1.15 * P.vf[jj[tg]])
        Qp[rr[tg], jj[tg]] = np.clip((gbq[tg] + gm["flow"].predict(Xg[tg], num_threads=8)) * P.lanes[jj[tg]], 60.0, None)
    Vb = np.maximum(Vp, Qp / P.kj[None, :])
    obs = np.isfinite(P.V) & np.isfinite(P.Q) & ~Pg.blank & ~Pg.target
    CV = complete(P.V / P.vf[None, :], obs, RANKS); CQ = complete(P.Q / P.cap[None, :], obs, RANKS)
    val = P.ramp_validity(); s0 = state_score(Pg, Vb, Qp); f0 = physics_score(Pg, Vb, Qp, validity=val)[2]
    sc = gap & Pg.target & np.isfinite(P.V) & np.isfinite(Vp)
    RM.setdefault("gapml", []).append(np.sqrt(np.mean((Vp[sc] - P.V[sc]) ** 2)))
    rr_, cc_ = np.nonzero(sc); kpos = np.zeros(P.T, int)
    for t, _ in wl:
        kpos[t + 1:t + G + 1] = np.arange(1, G + 1)
    CELLS.append(np.column_stack([np.full(len(rr_), pi), kpos[rr_], P.V[sc], Vp[sc], CV[0][sc] * P.vf[cc_], CV[8][sc] * P.vf[cc_], P.vf[cc_], P.Q[sc], Qp[sc], CQ[8][sc] * P.cap[cc_]]))
    for rk in RANKS:
        Vl = CV[rk] * P.vf[None, :]; Ql = CQ[rk] * P.cap[None, :]
        RM.setdefault(f"r{rk}", []).append(np.sqrt(np.nanmean((Vl[sc] - P.V[sc]) ** 2)))
        for w in WS:
            m = gap & np.isfinite(Vp) & np.isfinite(Vl) & np.isfinite(Ql)
            Vx, Qx = Vp.copy(), Qp.copy()
            Vx[m] = (1 - w) * Vp[m] + w * np.clip(Vl[m], 3.0, 1.15 * np.broadcast_to(P.vf, Vp.shape)[m])
            Qx[m] = (1 - w) * Qp[m] + w * np.clip(Ql[m], 60.0, None)
            Vx = np.maximum(Vx, Qx / P.kj[None, :])
            ds = state_score(Pg, Vx, Qx) - s0; dp = physics_score(Pg, Vx, Qx, validity=val)[2] - f0
            ACC.setdefault((rk, w), []).append((ds, dp))
    npn += 1; print(p, "windows", len(wl), "scored blank cells", int(sc.sum()), flush=True); load.cache_clear()

print("\nspeed RMSE on the scored blank cells (km/h, mean over panels):", {k: round(float(np.mean(v)), 2) for k, v in RM.items()})
print("blend inside the blanks (mean over panels): dS_state | dS_phys | 0.35 dS_state + 0.15 dS_phys")
for (rk, w), v in ACC.items():
    v = np.array(v); ds, dp = v.mean(0); se = v[:, 0].std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else np.nan
    print(f"  rank {rk} w {w}: {ds:+.5f} (SE {se:.5f}) | {dp:+.5f} | {0.35 * ds + 0.15 * dp:+.5f}")
np.save(WORK + "/b14_cells.npy", np.vstack(CELLS))   # panel, k, v_true, v_gapml, v_r0, v_r8, vf, q_true, q_gapml, q_r8
