"""G134 (10/31, ledger #94, strategy P9): the Feb harness of B26 (f3big + gapml_cv, both trained without February; 10 simulated T2 blanks per
T2 panel; density clip) with the gap cells blended with review/t1_gapnet.py: (1 - w) gapml + w net, w = 0.3 (declared), 0.5, 1 (record).
  data : writes the panel arrays and the window lists for the WSL trainer (C:/tfb_data/tfb_nn/gapnet/)
  eval : per panel dS_state, dS_phys and 0.35 dS_state + 0.15 dS_phys against the base; mean, SE over panels
usage: py g134_t1_gapnet.py [data | eval]"""
import sys, copy, os, pickle, time, numpy as np, lightgbm as lgb, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, r"C:/Users/user/tfb/src")
from cachelib import load, T2_PANELS
from common import WORK
from t1_features3 import grid_features3, rows3, base_of3
from t2_demand import profile
from t2_arrival import _dmin
from t1_gapml import view_train, gap_features, G
from metrics import state_score, physics_score
from common import PANELS
MODE = (sys.argv[1:] or ["data"])[0]; O = "C:/tfb_data/tfb_nn/gapnet/"; os.makedirs(O, exist_ok=True); MD = WORK + "/t1/"
Z = np.load(WORK + "/t2/gapml_windows.npz", allow_pickle=True); WM = Z["is_sel"]; t0 = time.time()


def feb_range(P):
    f = np.flatnonzero([s.startswith("2031-02") for s in P.times]); return f[0], f[-1] + 1


def harness_windows(p, a, T):
    return sorted({(int(t) - a, str(c)) for pp, t, c in zip(Z["panel"][WM], Z["t"][WM], Z["cond"][WM]) if str(pp) == p and 30 <= int(t) - a < T - G - 31})[:10]


if MODE == "data":
    L = {}
    for p in T2_PANELS:
        P = load(p, "train"); a, b = feb_range(P); fin = np.isfinite(P.V) & np.isfinite(P.Q) & P.E
        np.savez(O + p + ".npz", v=(P.V / P.vcut[None, :]).astype(np.float32), q=(P.Q / P.cap[None, :]).astype(np.float32), blank=P.blank, fin=fin)
        ts = np.array([int(t) for pp, t in zip(Z["panel"], Z["t"]) if str(pp) == p], int); ts = ts[(ts >= 30) & (ts < P.T - 40) & ~((ts >= a - 31) & (ts < b))]
        L["t_" + p] = ts; L["te_" + p] = np.array([t + a for t, _ in harness_windows(p, a, b - a)], int); print(p, len(ts), len(L["te_" + p]), f"{time.time() - t0:.0f}s", flush=True); load.cache_clear()
    np.savez(O + "windows.npz", panels=np.array(list(T2_PANELS)), **L); sys.exit()
mv = lgb.Booster(model_file=MD + "speed_f3big.txt"); mq = lgb.Booster(model_file=MD + "flow_f3big.txt"); gm = {ch: lgb.Booster(model_file=MD + f"gapml_{ch}_cv.txt") for ch in ("speed", "flow")}
PR = [pickle.load(open(O + f"pred_s{s}.pkl", "rb")) for s in (0, 1) if os.path.exists(O + f"pred_s{s}.pkl")]; print("seeds", len(PR), flush=True)
WS = (0.3, 0.5, 1.0); RES = {}; RMS = []


def feb(P):
    a, b = feb_range(P); Q = copy.copy(P)
    for k in ("times", "V", "Q", "O", "P", "E", "regime", "target", "blank", "R", "RE"):
        setattr(Q, k, getattr(P, k)[a:b].copy())
    Q.T = b - a; Q.day = np.array([t[:10] for t in Q.times[::288]]); return Q, a


for p in T2_PANELS:
    P0 = load(p, "train"); P, a0 = feb(P0); pi = PANELS.index(p); x = np.concatenate([[0.0], np.cumsum(P.length[:-1])]); dm = _dmin(P, x); prof = profile(P0, "2031-02"); wl = harness_windows(p, a0, P.T)
    Pg = copy.copy(P); Pg.blank = P.blank.copy(); Pg.target = P.target.copy(); gap = np.zeros((P.T, P.n), bool)
    for t, _ in wl:
        gap[t + 1:t + G + 1] = True; Pg.blank[t - 12:t] = False; Pg.target[t - 12:t] = False
    Pg.blank |= gap & np.isfinite(P.V)
    F, _, _, _ = grid_features3(Pg); bv, bq = base_of3(F); r, c = np.nonzero(Pg.target); X, _ = rows3(Pg, F, r, c); X = np.column_stack([np.full(len(r), pi, np.float32), X])
    V0 = np.full((P.T, P.n), np.nan); Q0 = V0.copy()
    V0[r, c] = np.clip((bv[r, c] + mv.predict(X, num_threads=16)) * P.vf[c], 3.0, 1.15 * P.vf[c]); Q0[r, c] = np.clip((bq[r, c] + mq.predict(X, num_threads=16)) * P.lanes[c], 60.0, None)
    Vn = V0.copy(); Qn = Q0.copy()
    for t, cnd in wl:
        V, Q = view_train(Pg, t); Xg, _, gbv, gbq, kk, jj = gap_features(Pg, V, Q, t, cnd, prof, dm, x); rr = t + kk; tg = Pg.target[rr, jj] & np.isfinite(gbv) & np.isfinite(gbq)
        V0[rr[tg], jj[tg]] = np.clip(gbv[tg] + gm["speed"].predict(Xg[tg], num_threads=16), 3.0, 1.15 * P.vf[jj[tg]]); Q0[rr[tg], jj[tg]] = np.clip((gbq[tg] + gm["flow"].predict(Xg[tg], num_threads=16)) * P.lanes[jj[tg]], 60.0, None)
        v_ = sum(d[(p, t + a0)][0] for d in PR) / len(PR); q_ = sum(d[(p, t + a0)][1] for d in PR) / len(PR)
        Vn[t + 1:t + 1 + G] = np.clip(v_ * P.vcut[None, :], 3.0, 1.15 * P.vf[None, :]); Qn[t + 1:t + 1 + G] = np.clip(q_ * P.cap[None, :], 60.0, None)
    tv = gap & Pg.target & np.isfinite(P.V) & np.isfinite(P.Q); RMS.append((p, float(np.sqrt(np.mean((V0[tv] - P.V[tv]) ** 2))), float(np.sqrt(np.mean((Vn[tv] - P.V[tv]) ** 2))), int(tv.sum())))
    val = P.ramp_validity(); Vb = np.maximum(V0, Q0 / P.kj[None, :]); s0 = (state_score(Pg, Vb, Q0), physics_score(Pg, Vb, Q0, validity=val)[2]); d = {}
    for w in WS:
        Vm = np.where(tv, (1 - w) * V0 + w * Vn, V0); Qm = np.where(tv, (1 - w) * Q0 + w * Qn, Q0); Vm = np.maximum(Vm, Qm / P.kj[None, :])
        s = (state_score(Pg, Vm, Qm), physics_score(Pg, Vm, Qm, validity=val)[2]); d[w] = (s[0] - s0[0], s[1] - s0[1])
    RES[p] = d; print(p, f"gap RMSE v {RMS[-1][1]:.2f} -> {RMS[-1][2]:.2f} | " + " | ".join(f"w {w}: dT1 {v[0]:+.4f} dT3 {v[1]:+.4f}" for w, v in d.items()), f"{time.time() - t0:.0f}s", flush=True); load.cache_clear()
print("\nmean over panels (SE):")
for w in WS:
    a = np.array([RES[p][w] for p in RES]); s = 0.35 * a[:, 0] + 0.15 * a[:, 1]; sd = lambda v: v.std(ddof=1) / np.sqrt(len(v))
    print(f"w {w}: dT1 {a[:, 0].mean():+.4f} (SE {sd(a[:, 0]):.4f}, z {a[:, 0].mean() / max(sd(a[:, 0]), 1e-9):+.1f}) | dT3 {a[:, 1].mean():+.4f} (SE {sd(a[:, 1]):.4f}) | 0.35 dT1 + 0.15 dT3 {s.mean():+.5f} (SE {sd(s):.5f}) | panels > 0: {(s > 0).sum()}/{len(s)}")
