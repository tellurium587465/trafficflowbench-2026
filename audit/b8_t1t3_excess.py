"""B8 (10/24, strategy (d)): where do Task 1 / Task 3 lose MORE than observation noise? Held-out February (f3big was trained
without it), the Feb harness of audit/b2 (10 selector-like windows per panel, rows T+5..T+90 blank on every link; f3big
on the Task 1 targets, the blank-cell model gapml_*_cv inside the blanks, density clip). No ADAPT / FD / boundary stages:
an approximation of the production T1.
Noise-floor reference: the observations are the state times white multiplicative noise (speed 0.8 %, flow 2.3 %, PLAN
§1.3), so a "prediction" equal to the observation times FRESH noise of the same size has the residual statistics of the
noise-free state. Per cell type: excess MSE = MSE(pred) - MSE(floor) (speed km/h, flow vph/lane) and excess conservation
residual = sum |dN_pred - dN_obs| - same for the floor, on the proxy S_LWR transitions (metrics.physics_score).
Types (separate partitions): T2 blank / within 3 rows of a blank / elsewhere; edge (observed <-> hidden) / interior
(hidden -> hidden); congested / free (observed speed vs v_cut); end link (first / last 2) / ramp link / plain. Also the
Jensen bias of k = q/v: mean(k_pred - k_obs) / mean(k_obs) by regime.
Pre-declared reading: >= 50 % of an excess in one type that holds <= 20 % of the cells -> that type is the target;
otherwise flat -> "T1 / T3 at the floor" with this as evidence.
usage: python b8_t1t3_excess.py [panels...]   (default: every corridor panel)"""
import sys, copy, numpy as np, lightgbm as lgb, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, r"C:/Users/user/tfb/src")
from cachelib import load
from common import PANELS, WORK
from t1_features3 import grid_features3, rows3, base_of3
from t2_demand import profile
from t2_arrival import _dmin
from t1_gapml import view_train, gap_features, G

MD = WORK + "/t1/"; rng = np.random.default_rng(8); SV, SQ = 0.008, 0.023
mv = lgb.Booster(model_file=MD + "speed_f3big.txt"); mq = lgb.Booster(model_file=MD + "flow_f3big.txt")
gm = {ch: lgb.Booster(model_file=MD + f"gapml_{ch}_cv.txt") for ch in ("speed", "flow")}
Z = np.load(WORK + "/t2/gapml_windows.npz", allow_pickle=True); WM = Z["is_sel"]


def feb(P):
    f = np.flatnonzero([s.startswith("2031-02") for s in P.times]); a, b = f[0], f[-1] + 1; Q = copy.copy(P)
    for k in ("times", "V", "Q", "O", "P", "E", "regime", "target", "blank", "R", "RE"):
        setattr(Q, k, getattr(P, k)[a:b].copy())
    Q.T = b - a; Q.day = np.array([t[:10] for t in Q.times[::288]])
    return Q, a


ACC = {}                                                         # (partition, type) -> [res_pred, res_floor, n_tr, se_v_p, se_v_f, se_q_p, se_q_f, n_cells]
JB = {"congested": [0.0, 0.0], "free": [0.0, 0.0]}
NOISE = {}


def add(part, typ, i, v):
    a = ACC.setdefault((part, typ), np.zeros(8)); a[i] += v


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
    for t, cnd in wl:                                              # the blank-cell model inside the blanks
        V, Q = view_train(Pg, t); Xg, _, gbv, gbq, kk, jj = gap_features(Pg, V, Q, t, cnd, prof, dm, x)
        rr = t + kk; tg = Pg.target[rr, jj] & np.isfinite(gbv) & np.isfinite(gbq)
        Vp[rr[tg], jj[tg]] = np.clip(gbv[tg] + gm["speed"].predict(Xg[tg], num_threads=8), 3.0, 1.15 * P.vf[jj[tg]])
        Qp[rr[tg], jj[tg]] = np.clip((gbq[tg] + gm["flow"].predict(Xg[tg], num_threads=8)) * P.lanes[jj[tg]], 60.0, None)
    Vp = np.maximum(Vp, Qp / P.kj[None, :])
    obs = np.isfinite(P.V) & np.isfinite(P.Q) & P.E
    T = Pg.target & obs & np.isfinite(Vp) & np.isfinite(Qp)        # scored target cells with a truth
    import os
    if os.environ.get("B8_NOISE") == "measured":                   # 10/29: noise measured from this panel's free flow
        okf = obs & (P.V > P.vcut[None, :] + 25); svl = []
        for j in range(P.n):
            v = P.V[okf[:, j], j]
            if len(v) > 300:
                m_ = np.median(v); svl.append(v[np.abs(v - m_) < 8].std())
        sv_abs = float(np.median(svl))
        noramp = ~np.r_[P.on_attached[1:] | P.off_attached[1:], True]                # link j and j+1 with no ramp at j+1
        pair = okf[:, :-1] & okf[:, 1:] & noramp[None, :-1]
        rel = ((P.Q[:, 1:] - P.Q[:, :-1]) / np.maximum(0.5 * (P.Q[:, 1:] + P.Q[:, :-1]), 1))[pair]; rel = rel[np.abs(rel) < 0.3]
        sq_rel = float(np.std(rel) / np.sqrt(2)); NOISE[p] = (sv_abs, sq_rel)
        sv_abs *= float(os.environ.get("B8_VSCALE", "1"))      # 10/29 B18: x sqrt(1 - 0.33) = the idiosyncratic part only (the per-row
        NOISE[p] = (sv_abs, sq_rel)                                # corridor-wide speed shock is readable from the visible cells of the row)
        Vf = P.V + sv_abs * rng.standard_normal(P.V.shape); Qf = P.Q * (1 + sq_rel * rng.standard_normal(P.Q.shape))
    else:
        Vf = P.V * (1 + SV * rng.standard_normal(P.V.shape)); Qf = P.Q * (1 + SQ * rng.standard_normal(P.Q.shape))   # floor reference
    lanes = P.lanes[None, :]
    near = np.zeros_like(gap)
    for d in (1, 2, 3):
        near[d:] |= gap[:-d]; near[:-d] |= gap[d:]
    near &= ~gap
    cong = obs & (P.V <= P.vcut[None, :])
    endl = np.zeros(P.n, bool); endl[:2] = endl[-2:] = True
    li = {l: i for i, l in enumerate(P.links)}; rampl = np.zeros(P.n, bool)
    rampl[[li[str(l)] for l in P.ramp_link if str(l) in li]] = True
    free = obs & ~cong; idx = np.arange(P.n)[None, :]                                 # 10/24: distance (links) to the nearest free observed cell
    left = np.maximum.accumulate(np.where(free, idx, -10 ** 6), axis=1); right = np.minimum.accumulate(np.where(free, idx, 10 ** 6)[:, ::-1], axis=1)[:, ::-1]
    dq = np.minimum(idx - left, right - idx)
    qpos = np.where(~cong, "free", np.where(dq <= 1, "queue edge", np.where(dq <= 3, "queue near", "queue deep")))
    cell_types = {"qpos": qpos, "blank": np.where(gap, "T2 blank", np.where(near, "near blank", "elsewhere")),
                  "regime": np.where(cong, "congested", "free"),
                  "link": np.broadcast_to(np.where(endl, "end link", np.where(rampl, "ramp link", "plain"))[None, :], P.V.shape),
                  "cross": np.where(cong, "cong", "free").astype(object) + np.where(gap, "+blank", np.where(near, "+near", "+other")).astype(object)}   # 10/24 diagnostic cross split
    # Task 1 excess MSE per type
    ev_p, ev_f = (Vp - P.V) ** 2, (Vf - P.V) ** 2; eq_p, eq_f = ((Qp - P.Q) / lanes) ** 2, ((Qf - P.Q) / lanes) ** 2
    for part, ty in cell_types.items():
        for typ in np.unique(ty[T]):
            m = T & (ty == typ)
            add(part, typ, 3, ev_p[m].sum()); add(part, typ, 4, ev_f[m].sum()); add(part, typ, 5, eq_p[m].sum()); add(part, typ, 6, eq_f[m].sum()); add(part, typ, 7, m.sum())
    for typ, m in (("congested", T & cong), ("free", T & ~cong)):
        kp, ko = Qp[m] / np.maximum(Vp[m], 1), P.Q[m] / np.maximum(P.V[m], 1); JB[typ][0] += (kp - ko).sum(); JB[typ][1] += ko.sum()
    # Task 3 proxy transitions (as metrics.physics_score): residual of pred and of the floor reference
    on_v, off_v, _, _ = P.ramp_validity()
    for rg in (0, 1, 2):
        day = (P.regime == rg)[:, None]; field = P.E & day & np.isfinite(P.V) & np.isfinite(P.Q); tgt = Pg.target & field
        Nt = np.where(field, P.Q / np.maximum(P.V, 1.0) * P.length[None, :], np.nan)
        res = {}
        for nm, (VV, QQ) in (("p", (Vp, Qp)), ("f", (Vf, Qf))):
            Vx = np.where(field, np.where(tgt, VV, P.V), np.nan); Qx = np.where(field, np.where(tgt, QQ, P.Q), np.nan)
            ok = field & np.isfinite(Vx) & np.isfinite(Qx); N = Qx / np.maximum(Vx, 1.0) * P.length[None, :]
            tr = ok[1:] & ok[:-1] & on_v[:-1] & off_v[:-1] & (tgt[1:] | tgt[:-1])     # transitions touching a target cell
            res[nm] = (np.abs((N[1:] - N[:-1]) - (Nt[1:] - Nt[:-1])), tr)
        tr = res["p"][1] & res["f"][1]
        kind = np.where(tgt[1:] & tgt[:-1], "interior", "edge")
        tt = {"blank": np.where(gap[1:] | gap[:-1], "T2 blank", np.where(near[1:] | near[:-1], "near blank", "elsewhere")),
              "regime": np.where(cong[1:] | cong[:-1], "congested", "free"), "kind": kind,
              "link": cell_types["link"][1:],
              "qpos": np.where(np.isin(qpos[1:], ["queue edge", "queue near", "queue deep"]), qpos[1:], qpos[:-1]),
              "cross": np.where(cong[1:] | cong[:-1], "cong", "free").astype(object) + np.where(gap[1:] | gap[:-1], "+blank", np.where(near[1:] | near[:-1], "+near", "+other")).astype(object)}
        for part, ty in tt.items():
            for typ in np.unique(ty[tr]):
                m = tr & (ty == typ)
                add(part, typ, 0, res["p"][0][m].sum()); add(part, typ, 1, res["f"][0][m].sum()); add(part, typ, 2, m.sum())
    print(p, "windows", len(wl), "target cells", int(T.sum()), flush=True)
    load.cache_clear()

print("\nTask 3 proxy: conservation residual, excess over the noise-floor reference, by type (share of the total excess; share of transitions)")
for part in ("blank", "kind", "regime", "link", "cross", "qpos"):
    rows = {t: v for (pp, t), v in ACC.items() if pp == part and v[2] > 0}
    ex = {t: v[0] - v[1] for t, v in rows.items()}; tot, ntr = sum(ex.values()), sum(v[2] for v in rows.values())
    print(f"  {part:7s}: " + " | ".join(f"{t} {ex[t] / tot:+.0%} of excess, {rows[t][2] / ntr:.0%} of transitions (pred/floor {rows[t][0] / rows[t][1]:.2f})" for t in sorted(rows)))
print("Task 1: RMSE of pred vs the noise floor by type (speed km/h | flow vph/lane), share of the excess MSE, share of cells")
for part in ("blank", "regime", "link", "cross", "qpos"):
    rows = {t: v for (pp, t), v in ACC.items() if pp == part and v[7] > 0}
    exv = {t: v[3] - v[4] for t, v in rows.items()}; exq = {t: v[5] - v[6] for t, v in rows.items()}; n = sum(v[7] for v in rows.values())
    for t in sorted(rows):
        v = rows[t]
        print(f"  {part:7s} {t:11s}: speed {np.sqrt(v[3] / v[7]):.2f} vs floor {np.sqrt(v[4] / v[7]):.2f} ({exv[t] / sum(exv.values()):+.0%} of excess) | "
              f"flow {np.sqrt(v[5] / v[7]):.1f} vs {np.sqrt(v[6] / v[7]):.1f} ({exq[t] / sum(exq.values()):+.0%}) | cells {v[7] / n:.1%}")
print("Jensen bias of k = q/v (mean (k_pred - k_obs) / mean k_obs):", {t: f"{v[0] / v[1]:+.2%}" for t, v in JB.items()})

print("noise used (speed km/h abs, flow rel):", {k: (round(v[0], 2), f"{v[1]:.2%}") for k, v in NOISE.items()})
