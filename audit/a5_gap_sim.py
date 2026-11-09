"""Audit A5: what the 90-minute Task 2 gap costs Task 1, measured with the team's own model.
February (held out from model f3big) of a train panel: add val-like gaps (rows T+1..T+18 blank on every link) at
up to 10 selector-like Feb windows, rebuild the f3 features, predict every Feb Task 1 target with/without the gaps
and score with the exact local scorer."""
import sys, copy, numpy as np, lightgbm as lgb, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, r"C:/Users/user/tfb/src"); sys.path.insert(0, r"C:/Users/user/tfb/audit")
from cachelib import load
from common import PANELS
from t1_features3 import grid_features3, rows3, base_of3
from metrics import state_score
from npzmm import npz_memmap

MD = "C:/Users/user/tfb/work/t1/"
mv = lgb.Booster(model_file=MD + "speed_f3big.txt"); mq = lgb.Booster(model_file=MD + "flow_f3big.txt")
Z = np.load("C:/Users/user/tfb/work/t2/featcache2_noT.npz", allow_pickle=True)
meta = {k: Z[k] for k in ("panel", "t", "date", "is_sel", "cond")}


def feb_panel(P):
    feb = np.flatnonzero([s.startswith("2031-02") for s in P.times]); a, b = feb[0], feb[-1] + 1
    Q = copy.copy(P)
    for k in ("times", "V", "Q", "O", "P", "E", "regime", "target", "blank"):
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


panels = sys.argv[1:] or ["D7_I405_N", "D7_I210_E", "D12_I5_S"]
for p in panels:
    P0 = load(p, "train"); P, a = feb_panel(P0); pi = PANELS.index(p)
    ks = [k for k in np.flatnonzero((meta["panel"] == p) & meta["is_sel"]) if str(meta["date"][k]).startswith("2031-02")]
    ts = sorted({int(meta["t"][k]) - a for k in ks})[:10]
    gap = np.zeros((P.T, P.n), bool)
    for t in ts:
        gap[t + 1:t + 19] = True
    Va, Qa = predict(P, pi)
    Pg = copy.copy(P); Pg.blank = P.blank | (gap & np.isfinite(P.V))
    Vg, Qg = predict(Pg, pi)
    m = P.target & P.E & np.isfinite(P.V); mg = m & gap
    lanes = np.broadcast_to(P.lanes[None, :], P.V.shape)
    rm = lambda A, B, s: np.sqrt(np.mean((A[s] - B[s]) ** 2))
    cong = m & gap & (P.V < 0.6 * P.vf[None, :])
    print(f"{p}: {len(ts)} gaps, targets {m.sum()}, in gaps {mg.sum()} ({mg.sum()/m.sum():.3%}), congested in gaps {cong.sum()}")
    print(f"   gap targets  RMSE v  no-gap model {rm(Va, P.V, mg):.2f}  with gap {rm(Vg, P.V, mg):.2f} | q/lane {rm(Qa/lanes, P.Q/lanes, mg):.1f} -> {rm(Qg/lanes, P.Q/lanes, mg):.1f}")
    print(f"   other targets RMSE v {rm(Va, P.V, m & ~gap):.2f} -> {rm(Vg, P.V, m & ~gap):.2f}")
    sa, da = state_score(P, Va, Qa, detail=True); sg, dg = state_score(P, Vg, Qg, detail=True)
    print(f"   Feb S_state without gaps {sa:.4f}  with val-like gaps {sg:.4f}  delta {sg-sa:+.4f}")
    load.cache_clear()
