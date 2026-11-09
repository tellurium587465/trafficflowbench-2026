"""G245 (11/03, ledger #210, the strategy's (3) second step, GOAL_FIRST_PLACE 8.5: sigma_noise 24.0 is in the band 22-26 -> the
February harness only). Does a linear correction after Task 1 lower the flow error on the hidden cells? G168's February harness
(training split, February held out of every model; the views with the simulated Task 1 blanks and up to 10 T2 blanks per panel),
the production mix (1 - w) chain + w net with w = 0.7 outside the T2 blanks and 0.1 inside (t1_unet_apply.py 0.7 0.1), the net =
seeds 4, 5 (unet_s45). Features per hidden target cell (t, j): the upstream link's flow at t - round(L / v_f) rows and the downstream
link's flow at t, each the observation where that cell is visible in the view, else the mix; used as (neighbour - mix), split by
visibility (4 columns, no intercept). The cell's own occupancy is empty on Task 1 targets, and the occupancy speed equals the
observed speed within +-5 % (G244), so no occupancy column. Fitted pooled on odd February days, read on even days (and reversed).
Line (8.5): flow RMSE against the observation down by >= 3 % on the read half and not within its SE -> the T3 proxy next.
usage: py -B -X utf8 g245_t1_flow_postfix.py"""
import sys, copy, numpy as np
sys.path.insert(0, r"C:/Users/user/tfb/src")
from cachelib import load
from common import WORK, PANELS
from t1_gapml import G
O = "C:/tfb_data/tfb_nn/unet_h/"; Z = np.load(WORK + "/t2/gapml_windows.npz", allow_pickle=True); WM = Z["is_sel"]


def feb_range(P):
    f = np.flatnonzero([s.startswith("2031-02") for s in P.times]); return f[0], f[-1] + 1


def harness_windows(p, a, T):
    return sorted({(int(t) - a, str(c)) for pp, t, c in zip(Z["panel"][WM], Z["t"][WM], Z["cond"][WM]) if str(pp) == p and 30 <= int(t) - a < T - G - 31})[:10]


def feb(P):
    a, b = feb_range(P); Q = copy.copy(P)
    for k in ("times", "V", "Q", "O", "P", "E", "regime", "target", "blank", "R", "RE"):
        setattr(Q, k, getattr(P, k)[a:b].copy())
    Q.T = b - a; Q.day = np.array([t[:10] for t in Q.times[::288]]); return Q, a


def view(P0):                                                   # G168's view, unchanged
    P, a0 = feb(P0); p = P0.panel; wl = harness_windows(p, a0, P.T); Pg = copy.copy(P); Pg.blank = P.blank.copy(); Pg.target = P.target.copy(); gap = np.zeros((P.T, P.n), bool)
    for t, _ in wl:
        gap[t + 1:t + G + 1] = True; Pg.blank[t - 12:t] = False; Pg.target[t - 12:t] = False
    Pg.blank |= gap & np.isfinite(P.V); return P, Pg, gap, wl, a0


XS, YS, DS, BS = [], [], [], []; vis_share = []
for p in PANELS:
    P0 = load(p, "train"); P, Pg, gap, wl, a0 = view(P0); z = np.load(O + p + ".npz"); u = np.load(O + p + "_unet_s45.npz")
    V0, Q0 = z["V0"].astype(float), z["Q0"].astype(float); gap = z["gap"]; QU = np.clip(u["QU"].astype(float), 60.0, None)
    tv = Pg.target & np.isfinite(P.V) & np.isfinite(P.Q) & P.E & np.isfinite(V0); w = np.where(gap, 0.1, 0.7); Qm = np.where(tv, (1 - w) * Q0 + w * QU, Q0)
    vis = np.isfinite(P.Q) & ~Pg.target & ~Pg.blank; Qn = np.where(vis, P.Q, Qm)       # what is known at each cell: the observation if visible, else the mix
    r, c = np.nonzero(tv); X = np.zeros((len(r), 4)); sh = np.rint(P.length / np.maximum(P.vf, 1.0) * 12).astype(int)   # rows of 5 minutes
    up = c > 0; ru = np.clip(r[up] - sh[c[up] - 1], 0, P.T - 1); cu = c[up] - 1; du = Qn[ru, cu] - Qm[r[up], c[up]]; vu = vis[ru, cu]
    ok = np.isfinite(du); idx = np.flatnonzero(up)[ok]; X[idx, 0] = np.where(vu[ok], du[ok], 0); X[idx, 1] = np.where(vu[ok], 0, du[ok])
    dn = c < P.n - 1; dd = Qn[r[dn], c[dn] + 1] - Qm[r[dn], c[dn]]; vd = vis[r[dn], c[dn] + 1]
    ok = np.isfinite(dd); idx = np.flatnonzero(dn)[ok]; X[idx, 2] = np.where(vd[ok], dd[ok], 0); X[idx, 3] = np.where(vd[ok], 0, dd[ok])
    XS.append(X); YS.append(P.Q[r, c] - Qm[r, c]); DS.append(r // 288); BS.append(np.full(len(r), PANELS.index(p))); vis_share.append((p, len(r), float(vu.mean()) if up.any() else 0.0, float(vd.mean()) if dn.any() else 0.0))
    print(p, "target cells", len(r), f"upstream visible {vis_share[-1][2]:.2f} downstream visible {vis_share[-1][3]:.2f}", flush=True); load.cache_clear()
X, Y, D, B = np.vstack(XS), np.concatenate(YS), np.concatenate(DS), np.concatenate(BS); ok = np.isfinite(Y); X, Y, D, B = X[ok], Y[ok], D[ok], B[ok]
rng = np.random.default_rng(0)
for fit_par, nm in ((1, "fitted on odd days, read on even days (the line)"), (0, "fitted on even days, read on odd days (record)")):
    f = D % 2 == (1 - fit_par); rd = ~f; beta = np.linalg.lstsq(X[f], Y[f], rcond=None)[0]; e0 = Y[rd]; e1 = Y[rd] - X[rd] @ beta
    dd_ = D[rd]; days = np.unique(dd_); s0 = np.array([np.sum(e0[dd_ == d] ** 2) for d in days]); s1 = np.array([np.sum(e1[dd_ == d] ** 2) for d in days]); n_ = np.array([np.sum(dd_ == d) for d in days])
    rel = np.sqrt(s1.sum() / n_.sum()) / np.sqrt(s0.sum() / n_.sum()) - 1; bs = []
    for _ in range(2000):
        i = rng.integers(0, len(days), len(days)); bs.append(np.sqrt(s1[i].sum() / n_[i].sum()) / np.sqrt(s0[i].sum() / n_[i].sum()) - 1)
    pp = [np.sqrt(np.mean(e1[B[rd] == b] ** 2)) / np.sqrt(np.mean(e0[B[rd] == b] ** 2)) - 1 for b in range(len(PANELS)) if (B[rd] == b).any()]
    print(f"{nm}: beta (up visible, up hidden, down visible, down hidden) {np.round(beta, 3)} | flow RMSE {np.sqrt(s0.sum() / n_.sum()):.1f} -> {np.sqrt(s1.sum() / n_.sum()):.1f} veh/h, "
          f"relative {rel:+.2%} (SE {np.std(bs):.2%}, days {len(days)}) | panels lower {sum(x < 0 for x in pp)}/{len(pp)} | line (<= -3 % and beyond 1 SE): {rel <= -0.03 and rel + np.std(bs) < 0}", flush=True)
