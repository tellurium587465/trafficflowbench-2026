"""B19 (10/29, ledger #57): the corridor-wide per-row speed shock (B18: about 1/3 of the speed noise variance, link-to-link
noise correlation +0.33 at any distance) estimated from the visible free-flow cells of the same row and added to the hidden
free-flow cells. Feb harness of B14 / B8 (f3big + gapml_cv + density clip), all panels.
c_t = mean over the visible free-flow cells (V >= 0.85 vf, not blank / target) of V - median(V of the same link at the visible
rows t-3..t+3 other than t), residuals clipped at 6 km/h; hidden cells predicted in free flow (Vp >= 0.85 vf) in rows with
>= 20 visible free cells get Vp + gamma c_t; flow untouched; density clip after.
gamma: the slope of the base residual (truth - Vp) on c_t over the other panels (leave one panel out, primary) and 1.
Prints per panel dS_state, dS_phys and 0.35 dS_state + 0.15 dS_phys; mean and SE over panels.
usage: py b19_row_shock.py [panels...]
Orientation header kept from B14:
B14 (10/29, strategy lane 12 (b), orientation before the gapml feature): what does a low-rank completion of the month
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


ACC = {}; RM = {}; npn = 0; CELLS = []; STORE = {}
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
    Vb = np.maximum(Vp, Qp / P.kj[None, :])
    vis = np.isfinite(P.V) & ~Pg.blank & ~Pg.target & (P.V >= 0.85 * P.vf[None, :])
    Vv = np.where(vis, P.V, np.nan); sm = np.full_like(Vv, np.nan)
    stack = np.stack([np.roll(Vv, k, axis=0) for k in (-3, -2, -1, 1, 2, 3)])
    stack[:, :3] = np.nan; stack[:, -3:] = np.nan
    with np.errstate(all="ignore"):
        sm = np.nanmedian(stack, axis=0)
        r = np.clip(Vv - sm, -6, 6); nvis = np.isfinite(r).sum(1); c = np.where(nvis >= 20, np.nanmean(r, axis=1), np.nan)
    tgt = Pg.target & np.isfinite(Vp) & (Vp >= 0.85 * P.vf[None, :]) & np.isfinite(c)[:, None]
    sc = tgt & np.isfinite(P.V)
    rr_, cc_ = np.nonzero(sc); CELLS.append(np.column_stack([np.full(len(rr_), pi), (P.V - Vb)[sc], c[rr_]]))
    val = P.ramp_validity(); s0 = state_score(Pg, Vb, Qp); f0 = physics_score(Pg, Vb, Qp, validity=val)[2]
    STORE[p] = (Pg, Vb, Qp, c, tgt, val, s0, f0, pi)
    print(p, "windows", len(wl), "hidden free cells corrected", int(tgt.sum()), "| rows with c", int(np.isfinite(c).sum()), "| sd(c) %.2f km/h" % np.nanstd(c), flush=True)
    load.cache_clear()

C = np.vstack(CELLS); res = {}
for p, (Pg, Vb, Qp, c, tgt, val, s0, f0, pi) in STORE.items():
    o = C[C[:, 0] != pi]; g_loo = float(np.sum(o[:, 1] * o[:, 2]) / np.sum(o[:, 2] ** 2))
    for nm, g in (("loo", g_loo), ("one", 1.0)):
        Vx = np.where(tgt, Vb + g * np.nan_to_num(c)[:, None], Vb); Vx = np.maximum(Vx, Qp / Pg.kj[None, :])
        ds = state_score(Pg, Vx, Qp) - s0; dp = physics_score(Pg, Vx, Qp, validity=val)[2] - f0
        res.setdefault(nm, []).append((ds, dp, g))
    print(p, "gamma(loo) %.3f" % g_loo, " | ".join(f"{nm}: dS_state {v[-1][0]:+.5f} dS_phys {v[-1][1]:+.5f}" for nm, v in res.items()), flush=True)
for nm, v in res.items():
    v = np.array(v); obj = 0.35 * v[:, 0] + 0.15 * v[:, 1]
    print(f"{nm}: panels {len(v)} | dS_state {v[:, 0].mean():+.5f} | dS_phys {v[:, 1].mean():+.5f} | objective {obj.mean():+.5f} (SE {obj.std(ddof=1) / np.sqrt(len(v)):.5f}, min panel {obj.min():+.5f}) | gamma mean {v[:, 2].mean():.3f}")
