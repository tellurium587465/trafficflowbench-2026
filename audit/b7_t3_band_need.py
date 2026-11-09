"""B7 (10/24, independent review B-1, need test): is the Task 3 conservation residual concentrated in cells whose regime is
ambiguous (predicted speed near the congestion threshold), where a median-type estimate would beat the T1 mean?
Held-out February (f3big was trained without it), the official Task 1 mask of the train split, f3big predictions as in the
Feb harness (audit/b2), density clip. The proxy S_LWR of metrics.physics_score (same transitions as the official scorer,
right-hand side = observed dN). Each transition's residual |dN - dN_obs| is attributed to the band of the predicted speed
relative to v_cut at its target cell(s) (the nearer one): |v_hat - v_cut| < 5 / 5-15 / 15-30 / >= 30 km/h. Oracle bound:
the proxy S_LWR when the predicted cells of a band are replaced by the observed values.
Pre-declared reading: oracle gain of the near-threshold bands (< 15 km/h) below 0.02 in S_LWR (= 0.002 in S at weight
2/3 x 0.15) -> stop B-1; otherwise build a median-type estimate for those cells.
usage: python b7_t3_band_need.py [panels...]   (default: the three Feb-harness panels)"""
import sys, copy, numpy as np, lightgbm as lgb, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, r"C:/Users/user/tfb/src")
from cachelib import load
from common import PANELS, WORK
from t1_features3 import grid_features3, rows3, base_of3
from metrics import physics_score

MD = WORK + "/t1/"
mv = lgb.Booster(model_file=MD + "speed_f3big.txt"); mq = lgb.Booster(model_file=MD + "flow_f3big.txt")
BANDS = [(0, 5), (5, 15), (15, 30), (30, 1e9)]


def feb(P):
    f = np.flatnonzero([s.startswith("2031-02") for s in P.times]); a, b = f[0], f[-1] + 1; Q = copy.copy(P)
    for k in ("times", "V", "Q", "O", "P", "E", "regime", "target", "blank", "R", "RE"):
        setattr(Q, k, getattr(P, k)[a:b].copy())
    Q.T = b - a; Q.day = np.array([t[:10] for t in Q.times[::288]])
    return Q


def lwr_parts(P, Vp, Qp, val):
    """per-transition residual and denominator of the proxy S_LWR, with the target cells involved (regime-pooled)"""
    on_v, off_v, _, _ = val; out = []
    for r in (0, 1, 2):
        day = (P.regime == r)[:, None]; field = P.E & day & np.isfinite(P.V) & np.isfinite(P.Q); tgt = P.target & field
        V = np.where(field, np.where(tgt, Vp, P.V), np.nan); Q = np.where(field, np.where(tgt, Qp, P.Q), np.nan)
        ok = field & np.isfinite(V) & np.isfinite(Q)
        N = Q / np.maximum(V, 1.0) * P.length[None, :]; Nt = np.where(field, P.Q / np.maximum(P.V, 1.0) * P.length[None, :], np.nan)
        tr = ok[1:] & ok[:-1] & on_v[:-1] & off_v[:-1]
        out.append((np.abs((N[1:] - N[:-1]) - (Nt[1:] - Nt[:-1])), np.abs(Nt[1:] - Nt[:-1]), tr, tgt))
    return out


for p in sys.argv[1:] or ["D7_I405_N", "D7_I210_E", "D12_I5_S"]:
    P = feb(load(p, "train")); pi = PANELS.index(p); val = P.ramp_validity()
    F, _, _, _ = grid_features3(P); bv, bq = base_of3(F); r, c = np.nonzero(P.target)
    X, _ = rows3(P, F, r, c); X = np.column_stack([np.full(len(r), pi, np.float32), X])
    Vp = np.full((P.T, P.n), np.nan); Qp = Vp.copy()
    Vp[r, c] = np.clip((bv[r, c] + mv.predict(X, num_threads=8)) * P.vf[c], 3.0, 1.15 * P.vf[c])
    Qp[r, c] = np.clip((bq[r, c] + mq.predict(X, num_threads=8)) * P.lanes[c], 60.0, None)
    Vp = np.maximum(Vp, Qp / P.kj[None, :])
    gap = np.abs(Vp - P.vcut[None, :])                          # distance of the prediction from the threshold (km/h)
    s0 = physics_score(P, Vp, Qp, validity=val)[1]
    res_tot = den_tot = 0.0; by = {b: [0.0, 0.0, 0] for b in BANDS}
    for res, den, tr, tgt in lwr_parts(P, Vp, Qp, val):
        g2 = np.fmin(np.where(tgt[1:], gap[1:], np.inf), np.where(tgt[:-1], gap[:-1], np.inf))   # nearer target cell of the pair
        res_tot += res[tr].sum(); den_tot += den[tr].sum()
        for lo, hi in BANDS:
            m = tr & (g2 >= lo) & (g2 < hi); by[(lo, hi)][0] += res[m].sum(); by[(lo, hi)][1] += den[m].sum(); by[(lo, hi)][2] += int(m.sum())
    line = [f"{p}: proxy S_LWR {s0:.4f} | residual / denominator share by |v_hat - v_cut| band, oracle S_LWR gain:"]
    for lo, hi in BANDS:
        Vo, Qo = Vp.copy(), Qp.copy(); band = (gap >= lo) & (gap < hi) & P.target
        Vo[band], Qo[band] = P.V[band], P.Q[band]
        so = physics_score(P, np.where(np.isfinite(Vo), Vo, Vp), np.where(np.isfinite(Qo), Qo, Qp), validity=val)[1]
        line.append(f"  [{lo:g}, {hi if hi < 1e9 else 'inf'}) km/h: transitions {by[(lo, hi)][2]:,}, residual {by[(lo, hi)][0] / res_tot:.1%} / denominator {by[(lo, hi)][1] / den_tot:.1%}, oracle +{so - s0:.4f}")
    print("\n".join(line), flush=True)
    load.cache_clear()
