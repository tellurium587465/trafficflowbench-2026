"""B11 (10/24, literature scan 2-1 / 2-4 and the strategy's decomposition (1); user go-ahead 10/24): the Feb harness of B8
(f3big + gapml_cv inside simulated Task 2 blanks + density clip), all panels.
(1) proxy S_LWR excess over the noise-floor reference (observations x fresh noise) by run (full-link blank / long run > 2
    rows / short run 1-2 rows) x transition (edge / interior) x switch (the true regime of the link changes within +-2 rows /
    steady).
(2) 2-1: inside each simulated blank, per link whose predicted speed crosses v_cut once between the first and the last
    predicted row, read the trajectory as the cumulative switch probability F_t = (v_first - v_t) / (v_first - v_last)
    (monotonised), p_t = dF_t, t* = argmax p_t; step = v_first before t*, v_last from t*; v = (1 - theta) v_hat + theta step,
    flow unchanged, density clip after. theta in {0.25, 0.5 (pre-declared primary), 1}.
    Pre-declared line: 0.35 dS_state + 0.15 dS_phys >= +0.001 and dS_FD >= -0.0005 (proxy, pooled) -> one LB submission;
    otherwise close.
(3) 2-4: signed mean of log(k_pred / k_obs) on target cells by free / congested / inside a blank / within 2 rows of a
    switch; pre-declared: |bias| < 0.3 % everywhere -> close.
usage: python b11_step_blank.py [panels...]"""
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

MD = WORK + "/t1/"; rng = np.random.default_rng(11); SV, SQ = 0.008, 0.023; THETAS = (0.25, 0.5, 1.0)
mv = lgb.Booster(model_file=MD + "speed_f3big.txt"); mq = lgb.Booster(model_file=MD + "flow_f3big.txt")
gm = {ch: lgb.Booster(model_file=MD + f"gapml_{ch}_cv.txt") for ch in ("speed", "flow")}
Z = np.load(WORK + "/t2/gapml_windows.npz", allow_pickle=True); WM = Z["is_sel"]


def feb(P):
    f = np.flatnonzero([s.startswith("2031-02") for s in P.times]); a, b = f[0], f[-1] + 1; Q = copy.copy(P)
    for k in ("times", "V", "Q", "O", "P", "E", "regime", "target", "blank", "R", "RE"):
        setattr(Q, k, getattr(P, k)[a:b].copy())
    Q.T = b - a; Q.day = np.array([t[:10] for t in Q.times[::288]])
    return Q, a


def lwr_res(P, tgt_all, VV, QQ):
    """per-transition |dN - dN_obs| on the proxy transitions, regimes pooled, and the transition mask (touching targets)"""
    on_v, off_v, _, _ = P.ramp_validity(); RES = np.zeros((P.T - 1, P.n)); TR = np.zeros((P.T - 1, P.n), bool)
    for rg in (0, 1, 2):
        day = (P.regime == rg)[:, None]; field = P.E & day & np.isfinite(P.V) & np.isfinite(P.Q); tgt = tgt_all & field
        Vx = np.where(field, np.where(tgt, VV, P.V), np.nan); Qx = np.where(field, np.where(tgt, QQ, P.Q), np.nan)
        ok = field & np.isfinite(Vx) & np.isfinite(Qx); N = Qx / np.maximum(Vx, 1.0) * P.length[None, :]
        Nt = np.where(field, P.Q / np.maximum(P.V, 1.0) * P.length[None, :], np.nan)
        tr = ok[1:] & ok[:-1] & on_v[:-1] & off_v[:-1] & (tgt[1:] | tgt[:-1])
        RES[tr] = np.abs((N[1:] - N[:-1]) - (Nt[1:] - Nt[:-1]))[tr]; TR |= tr
    return RES, TR


ACC, BIAS, SCORE = {}, {}, {}
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
    obs = np.isfinite(P.V) & np.isfinite(P.Q) & P.E; T = Pg.target & obs & np.isfinite(Vp)
    cong = obs & (P.V <= P.vcut[None, :])
    sw = np.zeros_like(cong); ch = (cong[1:] != cong[:-1]) & obs[1:] & obs[:-1]   # switch within +-2 rows
    for d in range(-2, 2):                                        # sw[i] = a change between rows i+d and i+d+1, d = -2..1
        i0, i1 = max(0, -d), min(P.T, P.T - 1 - d)
        sw[i0:i1] |= ch[i0 + d:i1 + d]
    hid = Pg.target | Pg.blank; run = np.zeros(hid.shape, np.int32)
    for j in range(P.n):                                          # length of the hidden run containing each cell
        h = hid[:, j]; dd = np.diff(np.r_[0, h.astype(int), 0]); s_, e_ = np.flatnonzero(dd == 1), np.flatnonzero(dd == -1)
        for a_, b_ in zip(s_, e_):
            run[a_:b_, j] = b_ - a_
    # (1) decomposition
    Vf = P.V * (1 + SV * rng.standard_normal(P.V.shape)); Qf = P.Q * (1 + SQ * rng.standard_normal(P.Q.shape))
    Rp, TRp = lwr_res(P, Pg.target, Vp, Qp); Rf, TRf = lwr_res(P, Pg.target, Vf, Qf); tr = TRp & TRf
    rk = np.where(gap[1:] | gap[:-1], "blank", np.where(np.maximum(run[1:], run[:-1]) > 2, "long", "short"))
    kd = np.where(Pg.target[1:] & Pg.target[:-1], "interior", "edge"); sk = np.where(sw[1:] | sw[:-1], "switch", "steady")
    key = np.char.add(np.char.add(rk.astype(str), "/"), np.char.add(np.char.add(kd.astype(str), "/"), sk.astype(str)))
    for k in np.unique(key[tr]):
        m = tr & (key == k); a = ACC.setdefault(k, np.zeros(3)); a += (Rp[m].sum(), Rf[m].sum(), m.sum())
    # (3) ratio bias
    lr = np.log(np.maximum(Qp, 1e-6) / np.maximum(Vp, 1.0)) - np.log(np.maximum(P.Q, 1e-6) / np.maximum(P.V, 1.0))
    for nm, m in (("free", T & ~cong & ~gap), ("congested", T & cong & ~gap), ("blank", T & gap), ("near switch", T & sw)):
        b_ = BIAS.setdefault(nm, np.zeros(2)); b_ += (np.nansum(lr[m]), np.isfinite(lr[m]).sum())
    # (2) the step inside the blanks
    val = P.ramp_validity(); base = (state_score(Pg, Vp, Qp), physics_score(Pg, Vp, Qp, validity=val))
    for th in (0.0,) + THETAS:
        Vs = Vp.copy(); nlinks = 0
        if th > 0:
            for t, _ in wl:
                rows = np.arange(t + 1, t + G + 1)
                for j in range(P.n):
                    m = Pg.target[rows, j] & np.isfinite(Vp[rows, j])
                    if m.sum() < 4:
                        continue
                    rr = rows[m]; v = Vp[rr, j]; v0, v1 = v[0], v[-1]
                    if (v0 - P.vcut[j]) * (v1 - P.vcut[j]) >= 0 or abs(v0 - v1) < 5:
                        continue
                    Fc = np.clip((v0 - v) / (v0 - v1), 0, 1); Fc = np.maximum.accumulate(Fc)
                    pt = np.diff(np.r_[0.0, Fc]); ts = int(np.argmax(pt))
                    step = np.where(np.arange(len(rr)) < ts, v0, v1)
                    Vs[rr, j] = (1 - th) * v + th * step; nlinks += 1
            Vs = np.maximum(Vs, Qp / P.kj[None, :])
        s = state_score(Pg, Vs, Qp); ph = physics_score(Pg, Vs, Qp, validity=val)
        acc = SCORE.setdefault(th, np.zeros(5)); acc += (s - base[0], ph[0] - base[1][0], ph[1] - base[1][1], ph[2] - base[1][2], nlinks)
    print(p, "windows", len(wl), flush=True)
    load.cache_clear()

npn = len(sys.argv[1:] or PANELS)
print("\n(1) proxy S_LWR excess over the floor by run / transition / switch (share of the total excess, share of transitions, pred/floor)")
tot = sum(v[0] - v[1] for v in ACC.values()); ntr = sum(v[2] for v in ACC.values())
for k, v in sorted(ACC.items(), key=lambda kv: -(kv[1][0] - kv[1][1])):
    print(f"  {k:28s} {(v[0] - v[1]) / tot:+6.1%} of excess | {v[2] / ntr:6.2%} of transitions | ratio {v[0] / max(v[1], 1e-9):.2f}")
print("(3) signed mean log(k_pred / k_obs):", {k: f"{v[0] / max(v[1], 1):+.2%}" for k, v in BIAS.items()})
print("(2) the step inside the blanks (mean over panels; dS_state, dS_FD, dS_LWR, dS_phys, links touched; 0.35 dS_state + 0.15 dS_phys):")
for th, v in SCORE.items():
    v = v / npn; obj = 0.35 * v[0] + 0.15 * v[3]
    print(f"  theta {th}: dS_state {v[0]:+.5f} dS_FD {v[1]:+.5f} dS_LWR {v[2]:+.5f} dS_phys {v[3]:+.5f} | links {v[4] * npn:.0f} | objective {obj:+.5f}")
