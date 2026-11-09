"""C1: validation-like February simulation of the best Task 1 pipeline, per panel.

February of a train panel (held out from f3big and from gapml_*_cv), made to look like a test month:
  - Task 2 blanks: 10 selector-like Feb windows (T2 panels only): rows t+1..t+18 blank on every link, history rows
    t-12..t-1 unmasked and not Task 1 targets (as in the release)
  - per-link free-flow speed redraw: V *= 1 + eps_v * ramp(v/vf), eps_v ~ N(0, 1.05%), ramp = 0 below 0.6 vf, 1 above 0.9 vf
  - per-link flow-level redraw: Q *= 1 + eps_q, eps_q ~ N(0, 0.7%)
  (truth = the perturbed values, as in a test month where the truth is the noisy observation)
Pipeline variants: f3big with train statics (= no ADAPT) / own-month statics (ADAPT) / ADAPT with train statics in the
feature columns (F3) -> gapml in blanks -> speed ceiling clip -> FD blend a.  Saves per-panel arrays for c2.
usage: python c1_sim.py <panel> [<panel> ...]
"""
import sys, os, copy, pickle, time, numpy as np, lightgbm as lgb, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, r"C:/Users/user/tfb/src")
from cachelib import load, T2_PANELS
from common import PANELS
import t1_features2 as F2, t1_features3 as F3m
from t1_features import kernel_mean
from t1_features3 import grid_features3, rows3, base_of3
from metrics import state_score, physics_score
from t2_demand import profile
import t2_demand
from t2_arrival import _dmin
from t1_gapml import view_train, gap_features, G

MD = "C:/Users/user/tfb/work/t1/"; OUT = "C:/Users/user/tfb/audit/sim/"; os.makedirs(OUT, exist_ok=True)
mv = lgb.Booster(model_file=MD + "speed_f3big.txt"); mq = lgb.Booster(model_file=MD + "flow_f3big.txt")
gm = {ch: lgb.Booster(model_file=MD + f"gapml_{ch}_cv.txt") for ch in ("speed", "flow")}
Z = np.load("C:/Users/user/tfb/work/t2/featcache2_noT.npz", allow_pickle=True)
meta = {k: Z[k] for k in ("panel", "t", "date", "is_sel", "cond")}; del Z
ORIG_LS = F2.link_statics


def feb_panel(P):
    feb = np.flatnonzero([s.startswith("2031-02") for s in P.times]); a, b = feb[0], feb[-1] + 1
    Q = copy.copy(P)
    for k in ("times", "V", "Q", "O", "P", "E", "regime", "target", "blank", "R", "RE"):
        setattr(Q, k, getattr(P, k)[a:b].copy())
    Q.T = b - a; Q.day = np.array([t[:10] for t in Q.times[::288]])
    return Q, a


def own_statics(P):
    """link_statics(own=True) on the simulated month as released (Task 1 targets blank)."""
    V = np.where(P.blank, np.nan, P.V); Qv = np.where(P.blank, np.nan, P.Q)
    ok = np.isfinite(V) & P.E & (V > 0.9 * P.vf[None, :])
    vr = np.where(ok, V / P.vf[None, :], np.nan); vmed = np.nanmedian(vr[::3], axis=0)
    x = np.concatenate([[0.0], np.cumsum(P.length[:-1])]); sub = slice(0, None, 7)
    Q = np.where(ok, Qv, np.nan)[sub]; km, _ = kernel_mean(Q, ok[sub], x, 1.0, [0], 1e9, 1.0)
    qratio = np.nanmedian(Q / km, axis=0)
    return np.nan_to_num(vmed, nan=1.0).astype(np.float32), np.nan_to_num(qratio, nan=1.0).astype(np.float32)


def f3_predict(P, pi, stat=None, f3cols=None):
    f = (lambda panel, split="train": stat) if stat is not None else ORIG_LS
    F2.link_statics = f; F3m.link_statics = f
    try:
        F, _, _, _ = grid_features3(P)
    finally:
        F2.link_statics = ORIG_LS; F3m.link_statics = ORIG_LS
    bv, bq = base_of3(F)
    r, c = np.nonzero(P.target)
    X, names = rows3(P, F, r, c)
    if f3cols is not None:
        X[:, names.index("_vmed")] = f3cols[0][c]; X[:, names.index("_qratio")] = f3cols[1][c]
    X = np.column_stack([np.full(len(r), pi, np.float32), X])
    Vp = np.full((P.T, P.n), np.nan); Qp = Vp.copy()
    Vp[r, c] = np.clip((bv[r, c] + mv.predict(X, num_threads=8)) * P.vf[c], 3.0, 1.15 * P.vf[c])
    Qp[r, c] = np.clip((bq[r, c] + mq.predict(X, num_threads=8)) * P.lanes[c], 60.0, None)
    del F, X
    return np.maximum(Vp, Qp / P.kj[None, :]), Qp


def gapml(P, Vp, Qp, wl, prof, dm, x):
    V2, Q2 = Vp.copy(), Qp.copy()
    for t, c in wl:
        V, Q = view_train(P, t)
        X, _, bv, bq, kk, jj = gap_features(P, V, Q, t, c, prof, dm, x)
        rr = t + kk; tg = P.target[rr, jj] & np.isfinite(bv) & np.isfinite(bq)
        V2[rr[tg], jj[tg]] = np.clip(bv + gm["speed"].predict(X, num_threads=8), 3.0, 1.15 * P.vf[jj])[tg]
        Q2[rr[tg], jj[tg]] = np.clip((bq + gm["flow"].predict(X, num_threads=8)) * P.lanes[jj], 60.0, None)[tg]
    return np.maximum(V2, Q2 / P.kj[None, :]), Q2


def ceil_clip(P, V, Q, capv):
    return np.maximum(np.minimum(V, capv[None, :]), Q / P.kj[None, :]), Q


def fd(P, V, Q, a):
    if a == 0:
        return V, Q
    cong = V < 0.6 * P.vf[None, :]; k = np.maximum(P.kj[None, :] - Q / P.w[None, :], 1e-3)
    Vn = np.where(cong, (1 - a) * V + a * Q / k, V)
    return np.maximum(Vn, Q / P.kj[None, :]), Q


def total_like(P, V, Q, val):
    s = state_score(P, V, Q); ph = physics_score(P, V, Q, validity=val)
    return s, ph[1], ph[2]


for p in sys.argv[1:]:
    t0 = time.time()
    P0 = load(p, "train"); P, a = feb_panel(P0); pi = PANELS.index(p)
    train_rows = np.array([s[:7] != "2031-02" for s in P0.times])
    capv_train = np.nanmax(np.where(train_rows[:, None] & np.isfinite(P0.V), P0.V, np.nan), axis=0)
    del P0; load.cache_clear()
    rng = np.random.default_rng(PANELS.index(p) + 100)
    ev = rng.normal(0, 0.0105, P.n); eq = rng.normal(0, 0.007, P.n)
    ramp = np.clip((P.V / P.vf[None, :] - 0.6) / 0.3, 0, 1)
    P.V = (P.V * (1 + ev[None, :] * ramp)).astype(P.V.dtype); P.Q = (P.Q * (1 + eq[None, :])).astype(P.Q.dtype)
    x = np.concatenate([[0.0], np.cumsum(P.length[:-1])]); dm = _dmin(P, x)
    ks = [k for k in np.flatnonzero((meta["panel"] == p) & meta["is_sel"]) if str(meta["date"][k]).startswith("2031-02")]
    wl = sorted({(int(meta["t"][k]) - a, str(meta["cond"][k])) for k in ks})[:10] if p in T2_PANELS else []
    Pg = copy.copy(P); Pg.blank = P.blank.copy(); Pg.target = P.target.copy()
    gap = np.zeros((P.T, P.n), bool); near = np.zeros((P.T, P.n), bool)
    for t, _ in wl:
        gap[t + 1:t + G + 1] = True; near[max(t - 12, 0):t + 1] = True; near[t + G + 1:t + G + 13] = True
        Pg.blank[t - 12:t] = False; Pg.target[t - 12:t] = False
    near &= ~gap
    Pg.blank |= gap & np.isfinite(P.V)
    val = P.ramp_validity()
    rel = Pg.blank & ~gap                                           # released-view blanks (= Task 1 targets)
    capv = np.fmax(capv_train, np.nanmax(np.where(~Pg.blank & np.isfinite(Pg.V), Pg.V, np.nan), axis=0))
    st_train = ORIG_LS(p, "train"); st_own = own_statics(Pg)
    prof = profile(load(p, "train"), "2031-02"); load.cache_clear()
    res = {}; keep = {}
    for var, stat, f3c in (("noadapt", None, None), ("adapt", st_own, None), ("adapt_f3", st_own, st_train)):
        Vm, Qm = f3_predict(Pg, pi, stat, f3c)
        res[(var, "f3 as is")] = total_like(Pg, Vm, Qm, val)
        Vg, Qg = gapml(Pg, Vm, Qm, wl, prof, dm, x) if wl else (Vm, Qm)
        res[(var, "+gapml")] = total_like(Pg, Vg, Qg, val)
        Vc, Qc = ceil_clip(Pg, Vg, Qg, capv)
        res[(var, "+clip")] = total_like(Pg, Vc, Qc, val)
        for aa in (0.5, 0.75, 1.0):
            Vf, Qf = fd(Pg, Vc, Qc, aa)
            res[(var, f"+fd{aa}")] = total_like(Pg, Vf, Qf, val)
            if var == "adapt" and aa == 0.75:
                keep = dict(V=Vf, Q=Qf, Vc=Vc, Qc=Qc)
        print(p, var, {k[1]: tuple(round(v, 5) for v in r) for k, r in res.items() if k[0] == var}, f"{time.time()-t0:.0f}s", flush=True)
    np.savez_compressed(OUT + f"{p}.npz", V=keep["V"], Q=keep["Q"], Vc=keep["Vc"], Qc=keep["Qc"], TV=Pg.V, TQ=Pg.Q, target=Pg.target, E=Pg.E,
                        regime=Pg.regime, gap=gap, near=near, vf=Pg.vf, lanes=Pg.lanes, length=Pg.length, kj=Pg.kj, w=Pg.w, cap=Pg.cap, kc=Pg.kc,
                        on_v=val[0], off_v=val[1], blank=Pg.blank, wl=np.array([t for t, _ in wl]))
    pickle.dump(res, open(OUT + f"{p}_scores.pkl", "wb"))
    t2_demand._PROF.clear()
    print(p, "done", f"{time.time()-t0:.0f}s", flush=True)
