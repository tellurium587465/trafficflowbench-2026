"""B25 (10/29, strategy J2, ledger #62): how much of the T1 excess in the congested / blank target cells sits where the
same month shows congestion at the same place and time of day on OTHER days? Sizes the one untested legal T1 lever
(supervised fine-tuning inside the month: hide visible cells of the month and learn to restore them; T1 is offline, no T
limit). Feb harness of B8 / B21 (f3big on the Task 1 targets + gapml_cv inside 10 simulated blanks per panel + density
clip; February is out of sample for both; no ADAPT, which only moves the free-flow link statics). Floor reference =
observation + fresh noise with the noise measured per panel, speed part x 0.82 (B18 / B21 new floor).
Scope: scored T1 target cells that are congested (observed V <= v_cut) or inside a simulated T2 blank.
Recurring: on >= 1 other day of the month a visible cell (finite, not blank, not a target) of the same link within +-6 rows
(30 min) of the same time of day has V < 0.6 vf (>= 3 and >= 7 other days alongside, diagnostic).
Prints the share of the scope's excess MSE (pred - floor; speed primary, flow alongside) in recurring cells, and the S value
= 0.35 dS_state + 0.15 dS_phys (proxy) when the prediction is replaced by the floor reference (perfect up to the noise) in
the recurring scope cells, and in the whole scope; mean and SE over panels.
Line (#62): share (speed, >= 1 day) >= 50 % and S value of the recurring cells >= +0.001 -> test the fine-tuning on
February; else close.
usage: py b25_t1_recur_share.py [panels...]"""
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


def other_days(C):
    """C (T, n) bool. Per cell: the number of OTHER days with C at the same link within +-6 rows of the same time of day."""
    D = -(-C.shape[0] // 288); M = np.zeros((D * 288, C.shape[1]), bool); M[:C.shape[0]] = C
    M = M.reshape(D, 288, -1); Md = M.copy()
    for d in range(1, 7):
        Md[:, d:] |= M[:, :-d]; Md[:, :-d] |= M[:, d:]
    return (Md.sum(0)[None] - Md).reshape(D * 288, -1)[:C.shape[0]]


SETS = ("scope", "rec1", "rec3", "rec7", "cong", "cong_rec1", "blank", "blank_rec1")
EX = {s: np.zeros(3) for s in SETS}                                # excess speed MSE sum, excess flow MSE sum, cells
SV = {}
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
    okf = np.isfinite(P.V) & np.isfinite(P.Q) & P.E & (P.V > P.vcut[None, :] + 25); svl = []
    for j in range(P.n):
        v = P.V[okf[:, j], j]
        if len(v) > 300:
            m_ = np.median(v); svl.append(v[np.abs(v - m_) < 8].std())
    sv_abs = float(np.median(svl))
    noramp = ~np.r_[P.on_attached[1:] | P.off_attached[1:], True]
    pair = okf[:, :-1] & okf[:, 1:] & noramp[None, :-1]
    rel = ((P.Q[:, 1:] - P.Q[:, :-1]) / np.maximum(0.5 * (P.Q[:, 1:] + P.Q[:, :-1]), 1))[pair]; rel = rel[np.abs(rel) < 0.3]
    sq_rel = float(np.std(rel) / np.sqrt(2)); val = P.ramp_validity(); rng = np.random.default_rng(25)
    Vf = P.V + 0.82 * sv_abs * rng.standard_normal(P.V.shape); Qf = P.Q * (1 + sq_rel * rng.standard_normal(P.Q.shape))
    obs = np.isfinite(P.V) & np.isfinite(P.Q) & P.E
    T = Pg.target & obs & np.isfinite(Vb) & np.isfinite(Qp)
    cong = obs & (P.V <= P.vcut[None, :]); scope = T & (cong | gap)
    vis = np.isfinite(P.V) & ~Pg.blank & ~Pg.target
    nod = other_days(vis & (P.V < 0.6 * P.vf[None, :]))
    M = {"scope": scope, "rec1": scope & (nod >= 1), "rec3": scope & (nod >= 3), "rec7": scope & (nod >= 7),
         "cong": scope & cong, "cong_rec1": scope & cong & (nod >= 1), "blank": scope & gap, "blank_rec1": scope & gap & (nod >= 1)}
    lanes = P.lanes[None, :]
    ev = (Vb - P.V) ** 2 - (Vf - P.V) ** 2; eq = ((Qp - P.Q) / lanes) ** 2 - ((Qf - P.Q) / lanes) ** 2
    for s, m in M.items():
        EX[s] += [ev[m].sum(), eq[m].sum(), m.sum()]
    b0 = state_score(Pg, Vb, Qp); p0 = physics_score(Pg, Vb, Qp, validity=val)[2]; sv = {}
    for s in ("rec1", "scope"):
        Vx = np.where(M[s], Vf, Vb); Qx = np.where(M[s], Qf, Qp)
        sv[s] = 0.35 * (state_score(Pg, Vx, Qx) - b0) + 0.15 * (physics_score(Pg, Vx, Qx, validity=val)[2] - p0)
    SV[p] = sv
    print(p, "windows", len(wl), "| scope cells", int(scope.sum()), "| recurring (>= 1 day)", int(M["rec1"].sum()),
          "| S value: recurring", f"{sv['rec1']:+.4f}", "| whole scope", f"{sv['scope']:+.4f}", flush=True)
    load.cache_clear()

tot = EX["scope"]
print("\nshare of the scope's excess MSE (speed | flow) and of its cells:")
for s in SETS[1:]:
    ref = EX["cong"] if s == "cong_rec1" else (EX["blank"] if s == "blank_rec1" else tot)
    print(f"  {s:10s} speed {EX[s][0] / ref[0]:.1%} | flow {EX[s][1] / ref[1]:.1%} | cells {EX[s][2] / ref[2]:.1%}"
          + ("  (of the congested part)" if s == "cong_rec1" else ("  (of the blank part)" if s == "blank_rec1" else "")))
for s in ("rec1", "scope"):
    v = np.array([SV[p][s] for p in SV])
    print(f"S value (0.35 dS_state + 0.15 dS_phys, floor in the {s} cells): mean {v.mean():+.4f} (SE {v.std(ddof=1) / np.sqrt(len(v)):.4f}) over {len(v)} panels")
sh = EX["rec1"][0] / tot[0]; s1 = np.mean([SV[p]["rec1"] for p in SV])
print(f"line (#62): share {sh:.1%} (>= 50 %) and S value {s1:+.4f} (>= +0.001) ->", "test the fine-tuning on February" if sh >= 0.5 and s1 >= 0.001 else "close")
