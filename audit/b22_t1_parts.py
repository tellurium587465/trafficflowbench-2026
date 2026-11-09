"""B22 (10/29, strategy upper-bound audit item 3): the T1 components that had a T1-only LB pair, switched on / off on the
Feb harness with one yardstick (S_state, S_FD, proxy S_LWR, proxy S_phys): gapml (LB +0.0052, s17 -> s22: blank cells from
f3big only vs the blank-cell model, no FD) and the FD blend (LB +0.0010, s22ac -> s22acf: a = 0.75 on cells predicted below
0.6 vf, gapml on). ADAPT (+0.0040) acts on the test months' redrawn free-flow speeds and cannot be measured on February;
the boundary model (+0.0003) uses final models fitted with February (in-sample here) -> both left out.
Per pair: LB diff - 0.35 x dS_state = 0.15 x dT3(LB) against the local dS_phys proxy.
usage: py b22_t1_parts.py
Header kept from B21:
B21 (10/29, strategy upper-bound audit item 2): the T1 / T3 headroom in S units on the Feb harness (B19 / B8), the floor
reference = observation x fresh noise with the noise measured per panel (B8_NOISE=measured: speed spread in free flow, flow
from ramp-free adjacent links), the speed part either as measured (old floor) or x sqrt(1 - 0.33) (B18: the per-row
corridor shock is readable, only the idiosyncratic part is a floor). Prints S_state and the proxy S_LWR / S_phys of the
model and of both floors, pooled over panels.
usage: py b21_floor_bound.py
Header kept from B19:
B19 (10/29, ledger #57): the corridor-wide per-row speed shock (B18: about 1/3 of the speed noise variance, link-to-link
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
    F, _, _, _ = grid_features3(Pg); bv, bq = base_of3(F); r, c = np.nonzero(Pg.target)
    X, _ = rows3(Pg, F, r, c); X = np.column_stack([np.full(len(r), pi, np.float32), X])
    Vp = np.full((P.T, P.n), np.nan); Qp = Vp.copy()
    Vp[r, c] = np.clip((bv[r, c] + mv.predict(X, num_threads=8)) * P.vf[c], 3.0, 1.15 * P.vf[c])
    Qp[r, c] = np.clip((bq[r, c] + mq.predict(X, num_threads=8)) * P.lanes[c], 60.0, None)
    Vp0, Qp0 = Vp.copy(), Qp.copy()
    for t, cnd in wl:
        V, Q = view_train(Pg, t); Xg, _, gbv, gbq, kk, jj = gap_features(Pg, V, Q, t, cnd, prof, dm, x)
        rr = t + kk; tg = Pg.target[rr, jj] & np.isfinite(gbv) & np.isfinite(gbq)
        Vp[rr[tg], jj[tg]] = np.clip(gbv[tg] + gm["speed"].predict(Xg[tg], num_threads=8), 3.0, 1.15 * P.vf[jj[tg]])
        Qp[rr[tg], jj[tg]] = np.clip((gbq[tg] + gm["flow"].predict(Xg[tg], num_threads=8)) * P.lanes[jj[tg]], 60.0, None)
    def fd(Vx, Qx, a=0.75):
        cong = Vx < 0.6 * P.vf[None, :]; k = np.maximum(P.kj[None, :] - Qx / P.w[None, :], 1e-3)
        return np.where(cong & np.isfinite(Vx), (1 - a) * Vx + a * Qx / k, Vx)
    clip = lambda Vx, Qx: np.maximum(Vx, Qx / P.kj[None, :])
    val = P.ramp_validity(); sc = lambda Vx, Qx: (state_score(Pg, Vx, Qx), *physics_score(Pg, Vx, Qx, validity=val))
    V_nog, V_g = clip(Vp0, Qp0), clip(Vp, Qp)
    out = {"no_gapml": sc(V_nog, Qp0), "gapml": sc(V_g, Qp), "gapml_fd": sc(clip(fd(V_g, Qp), Qp), Qp),
           "gapml_fd050": sc(clip(fd(V_g, Qp, 0.5), Qp), Qp), "gapml_fd100": sc(clip(fd(V_g, Qp, 1.0), Qp), Qp)}   # 10/29 #65: the LB sweep
    RES[p] = out; print(p, {k: tuple(round(float(x), 4) for x in v) for k, v in out.items()}, flush=True); load.cache_clear()

R = {k: np.array([RES[p][k] for p in RES]) for k in ("no_gapml", "gapml", "gapml_fd", "gapml_fd050", "gapml_fd100")}
print("mean over panels (S_state, S_FD, S_LWR proxy, S_phys proxy):")
for k, v in R.items():
    print(f"  {k:9s} " + " ".join(f"{x:.4f}" for x in v.mean(0)))
for nm, a, b, lb in (("gapml", "no_gapml", "gapml", 0.0052), ("FD 0.75", "gapml", "gapml_fd", 0.0010),
                     ("FD 0.5 vs 0.75", "gapml_fd", "gapml_fd050", 0.00006), ("FD 1.0 vs 0.75", "gapml_fd", "gapml_fd100", -0.00021)):
    d = (R[b] - R[a]).mean(0); t3 = (lb - 0.35 * d[0]) / 0.15; loc = 0.35 * d[0] + 0.15 * d[3]
    print(f"{nm}: local dS_state {d[0]:+.4f} dS_FD {d[1]:+.4f} dS_LWR {d[2]:+.4f} dS_phys {d[3]:+.4f} | local total 0.35 dS + 0.15 dP {loc:+.5f} | "
          f"LB {lb:+.5f} -> implied dT3(LB) {t3:+.4f} (vs local dS_phys {d[3]:+.4f}) | LB / local total {lb / loc if abs(loc) > 1e-9 else float('nan'):.2f}")
