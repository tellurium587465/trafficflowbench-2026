"""Audit A5b: cost of the Task 2 gap on Task 1 AND the Task 3 proxy, and how much a trivial fallback recovers.
Same setup as a5 (Feb of a train panel, val-like gaps at 10 selector-like Feb windows, team model f3big).
Fallback: inside a gap, a target cell gets the per-link linear interpolation in time between the last released value
before the gap and the first released value after it (both channels)."""
import sys, copy, numpy as np, lightgbm as lgb, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, r"C:/Users/user/tfb/src"); sys.path.insert(0, r"C:/Users/user/tfb/audit")
from cachelib import load
from common import PANELS
from t1_features3 import grid_features3, rows3, base_of3
from t1_features2 import inputs2
from metrics import state_score, physics_score

MD = "C:/Users/user/tfb/work/t1/"
mv = lgb.Booster(model_file=MD + "speed_f3big.txt"); mq = lgb.Booster(model_file=MD + "flow_f3big.txt")
Z = np.load("C:/Users/user/tfb/work/t2/featcache2_noT.npz", allow_pickle=True)
meta = {k: Z[k] for k in ("panel", "t", "date", "is_sel")}


def feb_panel(P):
    feb = np.flatnonzero([s.startswith("2031-02") for s in P.times]); a, b = feb[0], feb[-1] + 1
    Q = copy.copy(P)
    for k in ("times", "V", "Q", "O", "P", "E", "regime", "target", "blank", "R", "RE"):
        setattr(Q, k, getattr(P, k)[a:b].copy())
    Q.T = b - a; Q.day = np.array([t[:10] for t in Q.times[::288]])
    return Q, a


def predict(P, pi):
    F, _, _, _ = grid_features3(P); bv, bq = base_of3(F)
    r, c = np.nonzero(P.target)
    X, _ = rows3(P, F, r, c); X = np.column_stack([np.full(len(r), pi, np.float32), X])
    Vp = np.full((P.T, P.n), np.nan); Qp = Vp.copy()
    Vp[r, c] = np.clip((bv[r, c] + mv.predict(X, num_threads=16)) * P.vf[c], 3.0, 1.15 * P.vf[c])
    Qp[r, c] = np.clip((bq[r, c] + mq.predict(X, num_threads=16)) * P.lanes[c], 60.0, None)
    return Vp, Qp


def lin_fill(Pg, Vp, Qp, ts):
    V, Q, ok = inputs2(Pg)                      # what a participant sees (gaps blank)
    Vf, Qf = Vp.copy(), Qp.copy()
    for t in ts:
        lo, hi = max(t - 12, 0), min(t + 31, Pg.T)
        for j in range(Pg.n):
            pr = [i for i in range(t, lo - 1, -1) if ok[i, j]][:1]; nx = [i for i in range(t + 19, hi) if ok[i, j]][:1]
            if not pr or not nx:
                continue
            a, b = pr[0], nx[0]
            for i in range(t + 1, t + 19):
                if Pg.target[i, j]:
                    w = (i - a) / (b - a)
                    Vf[i, j] = V[a, j] * (1 - w) + V[b, j] * w; Qf[i, j] = Q[a, j] * (1 - w) + Q[b, j] * w
    return Vf, Qf


for p in sys.argv[1:] or ["D7_I405_N", "D12_I5_S"]:
    P0 = load(p, "train"); P, a = feb_panel(P0); pi = PANELS.index(p)
    ks = [k for k in np.flatnonzero((meta["panel"] == p) & meta["is_sel"]) if str(meta["date"][k]).startswith("2031-02")]
    ts = sorted({int(meta["t"][k]) - a for k in ks})[:10]
    gap = np.zeros((P.T, P.n), bool)
    for t in ts:
        gap[t + 1:t + 19] = True
    Va, Qa = predict(P, pi)
    Pg = copy.copy(P); Pg.blank = P.blank | (gap & np.isfinite(P.V))
    Vg, Qg = predict(Pg, pi)
    Vl, Ql = lin_fill(Pg, Vg, Qg, ts)
    val = P.ramp_validity()
    for nm, (A, B) in (("no gaps (local CV view)", (Va, Qa)), ("val-like gaps, model", (Vg, Qg)), ("val-like gaps, linear fallback in gaps", (Vl, Ql))):
        s = state_score(P, A, B); ph = physics_score(P, A, B, validity=val)
        print(f"{p} {nm:42s} S_state {s:.4f}  S_FD {ph[0]:.4f}  S_LWR_proxy {ph[1]:.4f}  S_phys_proxy {ph[2]:.4f}", flush=True)
    load.cache_clear()
