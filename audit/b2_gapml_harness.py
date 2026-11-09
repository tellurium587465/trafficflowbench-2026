"""B2: Feb harness (as a5b, made closer to validation): 10 selector-like Feb windows per panel; history rows T-60..T-5
unmasked and not Task 1 targets, rows T+5..T+90 blank on every link. Task 1 targets predicted by f3big, then the
blank cells replaced by (a) nothing, (b) linear interpolation, (c) the blank-cell model (gapml_*_cv, trained Jun-Dec).
Density clip v >= q/k_jam on everything. Exact local scorer + physics proxy."""
import sys, copy, numpy as np, lightgbm as lgb, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, r"C:/Users/user/tfb/src")
from cachelib import load
from common import PANELS
from t1_features3 import grid_features3, rows3, base_of3
from metrics import state_score, physics_score
from t2_demand import profile
from t2_arrival import _dmin
from t1_gapml import view_train, gap_features, G

MD = "C:/Users/user/tfb/work/t1/"
mv = lgb.Booster(model_file=MD + "speed_f3big.txt"); mq = lgb.Booster(model_file=MD + "flow_f3big.txt")
gm = {ch: lgb.Booster(model_file=MD + f"gapml_{ch}_cv.txt") for ch in ("speed", "flow")}
Z = np.load("C:/Users/user/tfb/work/t2/featcache2_noT.npz", allow_pickle=True)
meta = {k: Z[k] for k in ("panel", "t", "date", "is_sel", "cond")}


def feb_panel(P):
    feb = np.flatnonzero([s.startswith("2031-02") for s in P.times]); a, b = feb[0], feb[-1] + 1
    Q = copy.copy(P)
    for k in ("times", "V", "Q", "O", "P", "E", "regime", "target", "blank", "R", "RE"):
        setattr(Q, k, getattr(P, k)[a:b].copy())
    Q.T = b - a; Q.day = np.array([t[:10] for t in Q.times[::288]])
    return Q, a


def f3(P, pi):
    F, _, _, _ = grid_features3(P); bv, bq = base_of3(F)
    r, c = np.nonzero(P.target)
    X, _ = rows3(P, F, r, c); X = np.column_stack([np.full(len(r), pi, np.float32), X])
    Vp = np.full((P.T, P.n), np.nan); Qp = Vp.copy()
    Vp[r, c] = np.clip((bv[r, c] + mv.predict(X, num_threads=8)) * P.vf[c], 3.0, 1.15 * P.vf[c])
    Qp[r, c] = np.clip((bq[r, c] + mq.predict(X, num_threads=8)) * P.lanes[c], 60.0, None)
    return Vp, Qp


def clip(P, V, Q):
    return np.maximum(V, Q / P.kj[None, :]), Q


for p in sys.argv[1:] or ["D7_I405_N", "D7_I210_E", "D12_I5_S"]:
    P0 = load(p, "train"); P, a = feb_panel(P0); pi = PANELS.index(p)
    x = np.concatenate([[0.0], np.cumsum(P.length[:-1])]); dm = _dmin(P, x); prof = profile(P0, "2031-02")
    ks = [k for k in np.flatnonzero((meta["panel"] == p) & meta["is_sel"]) if str(meta["date"][k]).startswith("2031-02")]
    wl = sorted({(int(meta["t"][k]) - a, str(meta["cond"][k])) for k in ks})[:10]
    Pg = copy.copy(P); Pg.blank = P.blank.copy(); Pg.target = P.target.copy()
    gap = np.zeros((P.T, P.n), bool)
    for t, _ in wl:
        gap[t + 1:t + G + 1] = True
        Pg.blank[t - 12:t] = False; Pg.target[t - 12:t] = False
    Pg.blank |= gap & np.isfinite(P.V)
    Vm, Qm = f3(Pg, pi)
    Vl, Ql, Vg, Qg = Vm.copy(), Qm.copy(), Vm.copy(), Qm.copy()
    for t, c in wl:
        V, Q = view_train(Pg, t)
        X, names, bv, bq, kk, jj = gap_features(Pg, V, Q, t, c, prof, dm, x)
        rr = t + kk; tg = Pg.target[rr, jj] & np.isfinite(bv) & np.isfinite(bq)
        lv = np.clip(bv, 3.0, 1.15 * P.vf[jj]); lq = np.clip(bq * P.lanes[jj], 60.0, None)
        gv = np.clip(bv + gm["speed"].predict(X, num_threads=8), 3.0, 1.15 * P.vf[jj]); gq = np.clip((bq + gm["flow"].predict(X, num_threads=8)) * P.lanes[jj], 60.0, None)
        Vl[rr[tg], jj[tg]] = lv[tg]; Ql[rr[tg], jj[tg]] = lq[tg]; Vg[rr[tg], jj[tg]] = gv[tg]; Qg[rr[tg], jj[tg]] = gq[tg]
    val = P.ramp_validity(); m = Pg.target & gap & P.E & np.isfinite(P.V)
    lanes = np.broadcast_to(P.lanes[None, :], P.V.shape)
    for nm, (A, B) in (("f3big as is", (Vm, Qm)), ("linear in blank", (Vl, Ql)), ("gapml in blank", (Vg, Qg))):
        A, B = clip(P, A, B)
        s = state_score(Pg, A, B); ph = physics_score(Pg, A, B, validity=val)
        rv = np.sqrt(np.mean((A[m] - P.V[m]) ** 2)); rq = np.sqrt(np.mean(((B[m] - P.Q[m]) / lanes[m]) ** 2))
        print(f"{p} {nm:16s} blank-target RMSE v {rv:6.2f} q/lane {rq:6.1f} | S_state {s:.4f}  S_LWR_proxy {ph[1]:.4f}  S_phys_proxy {ph[2]:.4f}", flush=True)
    load.cache_clear()
