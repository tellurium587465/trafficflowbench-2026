"""B27c (10/29, ledger #69 addendum): how much can the organizer flux's departure from the observed dN (eta: the exact
answer scores S_LWR 0.952, not 1.0, B27b) move the proxy's T1-variant differences? The B22 harness (variants as B27),
the official LWR term in numpy (B27: = official to 3e-7 when rhs = the observed dN), rhs = dN_obs + eta with eta scaled so
that the exact answer scores 0.952 on each panel x regime:
  A: independent Laplace on every scored transition
  B: independent Laplace on the transitions of ramp-attached links only (ramp-noise-like)
Prints per eta model the mean over panels of S_LWR per variant and the four B22 pair differences (3 eta draws).
usage: py b27c_eta_sensitivity.py [panels...]"""
import sys, copy, numpy as np, lightgbm as lgb, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, r"C:/Users/user/tfb/src")
from cachelib import load
from common import PANELS, WORK
from t1_features3 import grid_features3, rows3, base_of3
from t2_demand import profile
from t2_arrival import _dmin
from t1_gapml import view_train, gap_features, G

MD = WORK + "/t1/"
mv = lgb.Booster(model_file=MD + "speed_f3big.txt"); mq = lgb.Booster(model_file=MD + "flow_f3big.txt")
gm = {ch: lgb.Booster(model_file=MD + f"gapml_{ch}_cv.txt") for ch in ("speed", "flow")}
Z = np.load(WORK + "/t2/gapml_windows.npz", allow_pickle=True); WM = Z["is_sel"]
REGS = (0, 1, 2); TARGET = 0.952; NDRAW = 3


def feb(P):
    f = np.flatnonzero([s.startswith("2031-02") for s in P.times]); a, b = f[0], f[-1] + 1; Q = copy.copy(P)
    for k in ("times", "V", "Q", "O", "P", "E", "regime", "target", "blank", "R", "RE"):
        setattr(Q, k, getattr(P, k)[a:b].copy())
    Q.T = b - a; Q.day = np.array([t[:10] for t in Q.times[::288]])
    return Q, a


def lwr(Pg, on_v, off_v, Vp, Qp, etas):
    """{eta model: [S_LWR per draw]} averaged over regimes; eta model None = the proxy (rhs = dN_obs)"""
    out = {}
    for r in REGS:
        day = (Pg.regime == r)[:, None]; field = Pg.E & day & np.isfinite(Pg.V) & np.isfinite(Pg.Q)
        tgt = Pg.target & field; V = np.where(tgt, Vp, Pg.V); Q = np.where(tgt, Qp, Pg.Q)
        ok = field & np.isfinite(V) & np.isfinite(Q)
        N = Q / np.maximum(V, 1.0) * Pg.length[None, :]; Nt = Pg.Q / np.maximum(Pg.V, 1.0) * Pg.length[None, :]
        tr = ok[1:] & ok[:-1] & on_v[:-1] & off_v[:-1]
        d, dt = (N[1:] - N[:-1])[tr], (Nt[1:] - Nt[:-1])[tr]
        for nm, e in etas[r].items():
            rhs = dt + e; out.setdefault(nm, []).append(1 - min(1.0, np.abs(d - rhs).sum() / (np.abs(rhs).sum() + 1e-9)))
    return {k: float(np.mean(v)) for k, v in out.items()}


def make_etas(Pg, on_v, off_v, rng):
    """per regime: {name: eta on the scored transitions}, scaled so the exact answer (d = dN_obs) scores TARGET"""
    att = (Pg.on_attached | Pg.off_attached)[None, :]
    etas = {}
    for r in REGS:
        day = (Pg.regime == r)[:, None]; ok = Pg.E & day & np.isfinite(Pg.V) & np.isfinite(Pg.Q)
        tr = ok[1:] & ok[:-1] & on_v[:-1] & off_v[:-1]
        Nt = Pg.Q / np.maximum(Pg.V, 1.0) * Pg.length[None, :]; dt = (Nt[1:] - Nt[:-1])[tr]
        mask_b = np.broadcast_to(att, tr.shape)[tr]
        e = {"proxy": np.zeros(tr.sum())}
        for i in range(NDRAW):
            for nm, msk in (("A", np.ones(tr.sum(), bool)), ("B", mask_b)):
                z = rng.laplace(size=tr.sum()) * msk
                lo, hi = 0.0, 10 * np.abs(dt).mean() / max(np.abs(z).mean(), 1e-9)
                for _ in range(50):                                   # bisection on the scale: exact answer -> TARGET
                    c = (lo + hi) / 2; rhs = dt + c * z; s = 1 - np.abs(c * z).sum() / np.abs(rhs).sum()
                    lo, hi = (c, hi) if s > TARGET else (lo, c)
                e[f"{nm}{i}"] = c * z
        etas[r] = e
    return etas


RES = {}; rng = np.random.default_rng(27)
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
    Vp0, Qp0 = Vp.copy(), Qp.copy()
    for t, cnd in wl:
        V, Q = view_train(Pg, t); Xg, _, gbv, gbq, kk, jj = gap_features(Pg, V, Q, t, cnd, prof, dm, x)
        rr = t + kk; tg = Pg.target[rr, jj] & np.isfinite(gbv) & np.isfinite(gbq)
        Vp[rr[tg], jj[tg]] = np.clip(gbv[tg] + gm["speed"].predict(Xg[tg], num_threads=8), 3.0, 1.15 * P.vf[jj[tg]])
        Qp[rr[tg], jj[tg]] = np.clip((gbq[tg] + gm["flow"].predict(Xg[tg], num_threads=8)) * P.lanes[jj[tg]], 60.0, None)
    def fd(Vx, Qx, a=0.75):
        cong = Vx < 0.6 * P.vf[None, :]; k = np.maximum(P.kj[None, :] - Qx / P.w[None, :], 1e-3)
        return np.where(cong & np.isfinite(Vx), (1 - a) * Vx + a * Qx / k, Vx)
    clip = lambda Vx, Qx: np.maximum(Vx, Qx / P.kj[None, :])
    V_g = clip(Vp, Qp)
    VAR = {"no_gapml": (clip(Vp0, Qp0), Qp0), "gapml": (V_g, Qp), "gapml_fd": (clip(fd(V_g, Qp), Qp), Qp),
           "gapml_fd050": (clip(fd(V_g, Qp, 0.5), Qp), Qp), "gapml_fd100": (clip(fd(V_g, Qp, 1.0), Qp), Qp)}
    on_v, off_v, _, _ = P.ramp_validity(); etas = make_etas(Pg, on_v, off_v, rng)
    RES[p] = {nm: lwr(Pg, on_v, off_v, Vx, Qx, etas) for nm, (Vx, Qx) in VAR.items()}
    RES[p]["truth"] = lwr(Pg, on_v, off_v, P.V, P.Q, etas)
    print(p, " | ".join(f"{k}: proxy {v['proxy']:.4f} A0 {v['A0']:.4f} B0 {v['B0']:.4f}" for k, v in RES[p].items()), flush=True)
    load.cache_clear()

models = ["proxy"] + [f"{m}{i}" for m in "AB" for i in range(NDRAW)]
print("\nmean over panels, S_LWR:")
for k in ("truth", "no_gapml", "gapml", "gapml_fd", "gapml_fd050", "gapml_fd100"):
    print(f"  {k:12s} " + " ".join(f"{m} {np.mean([RES[p][k][m] for p in RES]):.4f}" for m in models))
print("pair differences dS_LWR (mean over panels):")
for nm, a, b in (("gapml", "no_gapml", "gapml"), ("FD 0.75", "gapml", "gapml_fd"), ("FD 0.5 vs 0.75", "gapml_fd", "gapml_fd050"), ("FD 1.0 vs 0.75", "gapml_fd", "gapml_fd100")):
    d = {m: np.mean([RES[p][b][m] - RES[p][a][m] for p in RES]) for m in models}
    print(f"  {nm:15s} proxy {d['proxy']:+.5f} | A " + " ".join(f"{d[f'A{i}']:+.5f}" for i in range(NDRAW)) + " | B " + " ".join(f"{d[f'B{i}']:+.5f}" for i in range(NDRAW))
          + f" | ratio A/proxy {np.mean([d[f'A{i}'] for i in range(NDRAW)]) / d['proxy'] if abs(d['proxy']) > 1e-12 else float('nan'):.2f}"
          + f" B/proxy {np.mean([d[f'B{i}'] for i in range(NDRAW)]) / d['proxy'] if abs(d['proxy']) > 1e-12 else float('nan'):.2f}")
