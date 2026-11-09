"""B10 (10/24, strategy: a 2-hour need test before any T1 retraining): does Newell's lower envelope of cumulative counts
classify the regime (free / congested) of hidden cells near the queue boundary better than the T1 model?
Held-out February harness as B8 (f3big + gapml_cv inside the simulated Task 2 blanks + density clip). Population: Task 1
target cells outside the blanks with an observed truth, within 3 links of the true queue boundary (true congested cells with
a free observed cell <= 3 links away, true free cells with a congested observed cell <= 3 links away).
Newell for a hidden cell j at row t: anchors u < j and d > j = the nearest observed (not target, not blank) links at row t
within 6 links. Counts start one hour earlier (t0 = t - 12): N_u(t0) = 0, N_j(t0) = -M(u..j), N_d(t0) = -M(u..d), with
M = vehicles between the link midpoints (sum of k L from observed or predicted states at t0). Counts then grow with the
flows at u and d (observed, else the T1 prediction) and the ramp flows between (observed, else 0). Free branch at j:
N_u(t - tau_f) + ramps(u..j), tau_f = distance / free-flow speed; congested branch: N_d(t - tau_w) + k_jam L(j..d) - ramps(j..d),
tau_w = distance / wave speed w. Newell: congested iff the congested branch is the lower one.
Pre-declared line (strategy 10/24): Newell is right on >= 20 % of the cells the model gets wrong AND those fixes are >= 2x
the cells where the model is right and Newell wrong -> retrain T1 with it as a feature; otherwise close.
usage: python b10_newell_boundary.py [panels...]"""
import sys, copy, numpy as np, lightgbm as lgb, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, r"C:/Users/user/tfb/src")
from cachelib import load
from common import PANELS, WORK
from t1_features3 import grid_features3, rows3, base_of3
from t2_demand import profile
from t2_arrival import _dmin
from t1_gapml import view_train, gap_features, G

MD = WORK + "/t1/"; DT = 5 / 60
mv = lgb.Booster(model_file=MD + "speed_f3big.txt"); mq = lgb.Booster(model_file=MD + "flow_f3big.txt")
gm = {ch: lgb.Booster(model_file=MD + f"gapml_{ch}_cv.txt") for ch in ("speed", "flow")}
Z = np.load(WORK + "/t2/gapml_windows.npz", allow_pickle=True); WM = Z["is_sel"]


def feb(P):
    f = np.flatnonzero([s.startswith("2031-02") for s in P.times]); a, b = f[0], f[-1] + 1; Q = copy.copy(P)
    for k in ("times", "V", "Q", "O", "P", "E", "regime", "target", "blank", "R", "RE"):
        setattr(Q, k, getattr(P, k)[a:b].copy())
    Q.T = b - a; Q.day = np.array([t[:10] for t in Q.times[::288]])
    return Q, a


def interp_rows(C, x):
    """C: cumulative series on integer rows (index 0 = t0); value at fractional row x (clipped)"""
    x = np.clip(x, 0, len(C) - 1); i = int(np.floor(x)); f = x - i
    return C[i] if i + 1 >= len(C) else C[i] * (1 - f) + C[i + 1] * f


TOT = dict(n=0, model_wrong=0, fix=0, reverse=0, both_wrong=0, model_right=0, newell_right=0)
ZALL = dict(n=0, right=0, cong_as_free=0, free_as_cong=0, abs_err_right=0.0, abs_err_wrong=0.0)
for p in (sys.argv[1:] or PANELS):
    P0 = load(p, "train"); P, a0 = feb(P0); pi = PANELS.index(p)
    x = np.concatenate([[0.0], np.cumsum(P.length[:-1])]); dm = _dmin(P, x); prof = profile(P0, "2031-02"); xm = x + P.length / 2
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
    anchor = np.isfinite(P.V) & np.isfinite(P.Q) & P.E & ~Pg.target & ~Pg.blank
    Vm = np.where(anchor, P.V, Vp); Qm = np.where(anchor, P.Q, Qp); km = Qm / np.maximum(Vm, 1.0)
    li = {l: i for i, l in enumerate(P.links)}; rl = np.array([li.get(str(l), -1) for l in P.ramp_link]); sgn = np.where(P.ramp_type == "OR", 1.0, -1.0)
    R = np.nan_to_num(P.R.astype(float)) * sgn[None, :]
    cong_t = np.isfinite(P.V) & P.E & (P.V <= P.vcut[None, :]); free_t = np.isfinite(P.V) & P.E & ~cong_t
    idx = np.arange(P.n)[None, :]
    def near(mask):                                               # links to the nearest cell of mask in the same row
        left = np.maximum.accumulate(np.where(mask, idx, -10 ** 6), axis=1); right = np.minimum.accumulate(np.where(mask, idx, 10 ** 6)[:, ::-1], axis=1)[:, ::-1]
        return np.minimum(idx - left, right - idx)
    zone = Pg.target & ~gap & np.isfinite(P.V) & np.isfinite(Vp) & ((cong_t & (near(free_t & anchor) <= 3)) | (free_t & (near(cong_t & anchor) <= 3)))
    zone[:13] = False
    mz = Vp[zone] <= P.vcut[np.nonzero(zone)[1]]; tz = cong_t[zone]                 # 10/24: the model on the whole zone
    ZALL["n"] += int(zone.sum()); ZALL["right"] += int((mz == tz).sum()); ZALL["cong_as_free"] += int((tz & ~mz).sum()); ZALL["free_as_cong"] += int((~tz & mz).sum())
    ae = np.abs(Vp[zone] - P.V[zone]); ZALL["abs_err_right"] += float(ae[mz == tz].sum()); ZALL["abs_err_wrong"] += float(ae[mz != tz].sum())
    S = dict(n=0, model_wrong=0, fix=0, reverse=0, both_wrong=0, model_right=0, newell_right=0)
    for t, j in zip(*np.nonzero(zone)):
        us = [u for u in range(j - 1, max(-1, j - 7), -1) if anchor[t, u]]; ds = [d for d in range(j + 1, min(P.n, j + 7)) if anchor[t, d]]
        if not us or not ds:
            continue
        u, d = us[0], ds[0]; t0 = t - 12; rows = slice(t0, t + 1)
        if not (np.isfinite(km[t0, u:d + 1]).all() and np.isfinite(Qm[rows, u]).all() and np.isfinite(Qm[rows, d]).all()):
            continue
        Lm = lambda a, b: 0.5 * km[t0, a] * P.length[a] + km[t0, a + 1:b] @ P.length[a + 1:b] + 0.5 * km[t0, b] * P.length[b]
        Cu = np.r_[0.0, np.cumsum(Qm[t0:t, u] * DT)]; Cd = np.r_[0.0, np.cumsum(Qm[t0:t, d] * DT)] - Lm(u, d)
        ramp_uj = sum(R[t0:t, k].sum() * DT for k in np.flatnonzero((rl > u) & (rl <= j)))
        ramp_jd = sum(R[t0:t, k].sum() * DT for k in np.flatnonzero((rl > j) & (rl <= d)))
        vf = P.vf[u:j + 1].mean(); w = P.w[j:d + 1].mean()
        tf = 12 - (xm[j] - xm[u]) / max(vf, 1) / DT; tw = 12 - (xm[d] - xm[j]) / max(w, 1) / DT   # fractional row index from t0
        Nf = interp_rows(Cu, tf) + ramp_uj
        Nc = interp_rows(Cd, tw) + P.kj[j:d + 1].mean() * (xm[d] - xm[j]) - ramp_jd
        newell = Nc < Nf; model = Vp[t, j] <= P.vcut[j]; truth = cong_t[t, j]
        S["n"] += 1; S["model_right"] += model == truth; S["newell_right"] += newell == truth
        if model != truth:
            S["model_wrong"] += 1; S["fix"] += newell == truth; S["both_wrong"] += newell != truth
        elif newell != truth:
            S["reverse"] += 1
    print(p, S, flush=True)
    for k in TOT:
        TOT[k] += S[k]
    load.cache_clear()

n = max(TOT["n"], 1)
print(f"\nboundary-zone hidden cells {TOT['n']:,}: model regime right {TOT['model_right'] / n:.1%}, Newell right {TOT['newell_right'] / n:.1%}")
print(f"model wrong {TOT['model_wrong']:,}: Newell fixes {TOT['fix']:,} ({TOT['fix'] / max(TOT['model_wrong'], 1):.1%}); model right but Newell wrong {TOT['reverse']:,}")
ok = TOT["fix"] >= 0.2 * TOT["model_wrong"] and TOT["fix"] >= 2 * TOT["reverse"]
print("pre-declared line (fixes >= 20 % of the model's errors and >= 2x the reverse):", "PASS -> retrain T1 with it" if ok else "not met -> close")
z = ZALL; nz = max(z["n"], 1)
print(f"whole boundary zone (no anchor condition) {z['n']:,} cells: model regime right {z['right'] / nz:.1%} (congested predicted free {z['cong_as_free']:,}, free predicted congested {z['free_as_cong']:,}); "
      f"share of the zone's absolute speed error on regime-wrong cells {z['abs_err_wrong'] / max(z['abs_err_right'] + z['abs_err_wrong'], 1e-9):.1%}")
