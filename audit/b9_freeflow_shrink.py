"""B9 (10/24, strategy step 1 after B8): does the Task 1 model add variance in free flow that the month's own free-flow speed
would remove? In free flow the speed of a link is its free-flow speed times white noise (report 2.1), so the best estimate
of a hidden free-flow cell is the link's free-flow speed for the month, estimated from that month's observed free-flow
cells (allowed for Task 1, an offline task; the release shows the whole month).
Held-out February harness as B8 (f3big + gapml_cv inside the simulated Task 2 blanks + density clip; no ADAPT / FD /
boundary stages). Confident free flow (pre-declared): predicted speed >= max(v_cut + 15, 0.9 x the link's month free-flow
speed). Pre-declared variant (k = 1): the speed of those target cells replaced by the month free-flow speed (alpha = 0);
alpha = 0.25 / 0.5 / 0.75 blends are reported as a curve only. Flow is left as predicted.
Reading: S_state + proxy S_LWR (as metrics) on all target cells, per panel and pooled; line for a Leaderboard test:
point gain in S_state and no loss in S_LWR on >= 7 of 10 panels.
usage: python b9_freeflow_shrink.py [panels...]"""
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
ALPHAS = (0.0, 0.25, 0.5, 0.75)


def feb(P):
    f = np.flatnonzero([s.startswith("2031-02") for s in P.times]); a, b = f[0], f[-1] + 1; Q = copy.copy(P)
    for k in ("times", "V", "Q", "O", "P", "E", "regime", "target", "blank", "R", "RE"):
        setattr(Q, k, getattr(P, k)[a:b].copy())
    Q.T = b - a; Q.day = np.array([t[:10] for t in Q.times[::288]])
    return Q, a


rows = []
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
    Vp = np.maximum(Vp, Qp / P.kj[None, :])
    # the month's free-flow speed per link from the observed (released, not target) free-flow cells
    seen = np.isfinite(P.V) & ~Pg.target & ~Pg.blank & P.E & (P.V > P.vcut[None, :] + 15)
    vfm = np.array([np.nanmedian(P.V[seen[:, j], j]) if seen[:, j].sum() >= 50 else np.nan for j in range(P.n)])
    conf = Pg.target & np.isfinite(Vp) & np.isfinite(vfm)[None, :] & (Vp >= np.maximum(P.vcut + 15, 0.9 * np.nan_to_num(vfm, nan=1e9))[None, :])
    val = P.ramp_validity(); base = (state_score(Pg, Vp, Qp), physics_score(Pg, Vp, Qp, validity=val)[1])
    obs = conf & np.isfinite(P.V); rm = lambda A: float(np.sqrt(np.mean((A[obs] - P.V[obs]) ** 2)))
    out = dict(panel=p, conf_share=conf.sum() / max(Pg.target.sum(), 1), rmse_pred=rm(Vp), S_state=base[0], S_lwr=base[1])
    for a in ALPHAS:
        Va = Vp.copy(); Va[conf] = a * Vp[conf] + (1 - a) * np.broadcast_to(vfm[None, :], Vp.shape)[conf]
        Va = np.maximum(Va, Qp / P.kj[None, :])
        out[f"rmse_a{a}"] = rm(Va); out[f"dS_state_a{a}"] = state_score(Pg, Va, Qp) - base[0]; out[f"dS_lwr_a{a}"] = physics_score(Pg, Va, Qp, validity=val)[1] - base[1]
    rows.append(out); print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in out.items()}, flush=True)
    load.cache_clear()

import pandas as pd
D = pd.DataFrame(rows)
print("\npooled (mean over panels):")
for a in ALPHAS:
    print(f"  alpha {a}: speed RMSE in confident free cells {D[f'rmse_a{a}'].mean():.3f} (pred {D.rmse_pred.mean():.3f}) | dS_state {D[f'dS_state_a{a}'].mean():+.5f} "
          f"(panels > 0: {(D[f'dS_state_a{a}'] > 0).sum()}/{len(D)}) | dS_LWR {D[f'dS_lwr_a{a}'].mean():+.5f} (panels >= 0: {(D[f'dS_lwr_a{a}'] >= 0).sum()}/{len(D)})")
print(f"confident free share of the target cells {D.conf_share.mean():.1%}")
