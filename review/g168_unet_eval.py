"""G168 (11/01, ledger #118): the Feb harness of B26 / G134 on all 10 panels (f3big + gapml_cv, both trained without February; up to 10 simulated T2 blanks per T2 panel; density clip) with the image
network of tfb-t1-unet.py mixed in on the target cells: (1 - w) base + w net, w_out outside the simulated blanks, w_in inside them.
  base : caches the base prediction and the harness view per panel in C:/tfb_data/tfb_nn/unet_h/ (long part, CPU)
  infer: mean of the saved seeds on the harness view of February, C:/tfb_data/kaggle_out/t1unet/unet_s*.pt
  eval : grid w_out x w_in, chosen on odd days of February, read on even days (and the reverse as a record): 0.35 dS_state + 0.15 dS_phys, panels > 0, per panel
usage: py -B -X utf8 g168_unet_eval.py base | infer | eval"""
import sys, copy, os, glob, pickle, time, numpy as np, lightgbm as lgb, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, r"C:/Users/user/tfb/src")
from cachelib import load
from common import WORK, PANELS
from t1_features3 import grid_features3, rows3, base_of3
from t2_demand import profile
from t2_arrival import _dmin
from t1_gapml import view_train, gap_features, G
from metrics import state_score, physics_score

MODE = (sys.argv[1:] or ["base"])[0]; O = "C:/tfb_data/tfb_nn/unet_h/"; os.makedirs(O, exist_ok=True); MD = WORK + "/t1/"; KO = "C:/tfb_data/kaggle_out/t1unet/"
Z = np.load(WORK + "/t2/gapml_windows.npz", allow_pickle=True); WM = Z["is_sel"]; T00 = time.time()


def feb_range(P):
    f = np.flatnonzero([s.startswith("2031-02") for s in P.times]); return f[0], f[-1] + 1


def harness_windows(p, a, T):
    return sorted({(int(t) - a, str(c)) for pp, t, c in zip(Z["panel"][WM], Z["t"][WM], Z["cond"][WM]) if str(pp) == p and 30 <= int(t) - a < T - G - 31})[:10]


def feb(P):
    a, b = feb_range(P); Q = copy.copy(P)
    for k in ("times", "V", "Q", "O", "P", "E", "regime", "target", "blank", "R", "RE"):
        setattr(Q, k, getattr(P, k)[a:b].copy())
    Q.T = b - a; Q.day = np.array([t[:10] for t in Q.times[::288]]); return Q, a


def view(P0):
    P, a0 = feb(P0); p = P0.panel; wl = harness_windows(p, a0, P.T); Pg = copy.copy(P); Pg.blank = P.blank.copy(); Pg.target = P.target.copy(); gap = np.zeros((P.T, P.n), bool)
    for t, _ in wl:
        gap[t + 1:t + G + 1] = True; Pg.blank[t - 12:t] = False; Pg.target[t - 12:t] = False
    Pg.blank |= gap & np.isfinite(P.V); return P, Pg, gap, wl, a0


if MODE == "base":
    mv = lgb.Booster(model_file=MD + "speed_f3big.txt"); mq = lgb.Booster(model_file=MD + "flow_f3big.txt"); gm = {ch: lgb.Booster(model_file=MD + f"gapml_{ch}_cv.txt") for ch in ("speed", "flow")}
    for p in PANELS:
        if os.path.exists(O + p + ".npz"):
            continue
        P0 = load(p, "train"); P, Pg, gap, wl, a0 = view(P0); pi = PANELS.index(p); x = np.concatenate([[0.0], np.cumsum(P.length[:-1])])
        F, _, _, _ = grid_features3(Pg); bv, bq = base_of3(F); r, c = np.nonzero(Pg.target); X, _ = rows3(Pg, F, r, c); X = np.column_stack([np.full(len(r), pi, np.float32), X])
        V0 = np.full((P.T, P.n), np.nan); Q0 = V0.copy()
        V0[r, c] = np.clip((bv[r, c] + mv.predict(X, num_threads=8)) * P.vf[c], 3.0, 1.15 * P.vf[c]); Q0[r, c] = np.clip((bq[r, c] + mq.predict(X, num_threads=8)) * P.lanes[c], 60.0, None)
        if wl:
            dm = _dmin(P, x); prof = profile(P0, "2031-02")
            for t, cnd in wl:
                V, Q = view_train(Pg, t); Xg, _, gbv, gbq, kk, jj = gap_features(Pg, V, Q, t, cnd, prof, dm, x); rr = t + kk; tg = Pg.target[rr, jj] & np.isfinite(gbv) & np.isfinite(gbq)
                V0[rr[tg], jj[tg]] = np.clip(gbv[tg] + gm["speed"].predict(Xg[tg], num_threads=8), 3.0, 1.15 * P.vf[jj[tg]]); Q0[rr[tg], jj[tg]] = np.clip((gbq[tg] + gm["flow"].predict(Xg[tg], num_threads=8)) * P.lanes[jj[tg]], 60.0, None)
        np.savez(O + p + ".npz", V0=V0.astype(np.float32), Q0=Q0.astype(np.float32), gap=gap, nw=len(wl)); print(p, len(wl), f"{time.time() - T00:.0f}s", flush=True); load.cache_clear()
    sys.exit()


def net_src():
    s = open("C:/tfb_data/kaggle_up/k_t1unet/tfb-t1-unet.py", encoding="utf-8").read(); a = s.index("class Res(nn.Module)"); b = s.index("def load_all()"); return s[a:b]


if MODE == "infer":
    import torch, torch.nn as nn, torch.nn.functional as Fn
    exec(net_src()); torch.set_num_threads(8); fs = sorted(glob.glob(KO + "unet_s*.pt")); nets = []
    for f in fs:
        d = torch.load(f, map_location="cpu", weights_only=False); b = d["_base"]; print(f, "epoch", d["_ep"], flush=True); n = Net(14, b); n.load_state_dict({k: v.float() for k, v in d.items() if not k.startswith("_")}); n.eval(); nets.append(n)
    from_days = lambda dd: float(np.datetime64(str(dd)[:10]).astype("datetime64[D]").astype(int) % 7 in (2, 3))
    for p in PANELS:
        P0 = load(p, "train"); P, Pg, gap, wl, a0 = view(P0); stat = [P.vf / 120, P.lanes / 6, P.cap / 12000, np.minimum(P.length, 3) / 3, P.on_attached.astype(np.float32), P.off_attached.astype(np.float32)]; vis0 = np.isfinite(P.V) & np.isfinite(P.Q) & ~Pg.target & ~Pg.blank
        VU = np.zeros((P.T, P.n), np.float32); QU = VU.copy(); t1 = time.time()
        for i in range(P.T // 288):
            a, b = i * 288, (i + 1) * 288; x, fv, fq = image(p, P.V[a:b] / P.vf[None, :], P.Q[a:b] / P.cap[None, :], vis0[a:b], np.arange(288), stat, from_days(P.day[i]))
            with torch.no_grad():
                o = sum(n(pad8(torch.from_numpy(x)[None])[0])[..., :P.n] for n in nets) / len(nets)
            VU[a:b] = (o[0, 0].numpy() + fv) * P.vf; QU[a:b] = (o[0, 1].numpy() + fq) * P.cap
        np.savez(O + p + "_unet.npz", VU=VU, QU=QU); print(p, f"{(time.time() - t1) / (P.T // 288):.2f} s per day", flush=True); load.cache_clear()
    sys.exit()

WO = (0.1, 0.2, 0.3); WI = (0.0, 0.1, 0.2, 0.3); R = {}; RMS = []
for p in PANELS:
    P0 = load(p, "train"); P, Pg, gap, wl, a0 = view(P0); z = np.load(O + p + ".npz"); u = np.load(O + p + "_unet.npz"); V0, Q0 = z["V0"].astype(float), z["Q0"].astype(float); gap = z["gap"]
    VU = np.clip(u["VU"].astype(float), 3.0, 1.15 * P.vf[None, :]); QU = np.clip(u["QU"].astype(float), 60.0, None); val = P.ramp_validity(); tv = Pg.target & np.isfinite(P.V) & np.isfinite(P.Q) & P.E & np.isfinite(V0); day = np.arange(P.T) // 288; R[p] = {}
    RMS.append((p, float(np.sqrt(np.mean((V0[tv] - P.V[tv]) ** 2))), float(np.sqrt(np.mean((VU[tv] - P.V[tv]) ** 2))), float(np.corrcoef((V0 - P.V)[tv], (VU - P.V)[tv])[0, 1])))
    for par in (0, 1):                                            # 0 = odd calendar days (day index even)
        Pe = copy.copy(Pg); Pe.target = Pg.target & ((day % 2 == par)[:, None]); Vb = np.maximum(V0, Q0 / P.kj[None, :]); s0 = (state_score(Pe, Vb, Q0), physics_score(Pe, Vb, Q0, validity=val)[2])
        for wo in WO:
            for wi in WI:
                w = np.where(gap, wi, wo); Vm = np.where(tv, (1 - w) * V0 + w * VU, V0); Qm = np.where(tv, (1 - w) * Q0 + w * QU, Q0); Vm = np.maximum(Vm, Qm / P.kj[None, :]); s = (state_score(Pe, Vm, Qm), physics_score(Pe, Vm, Qm, validity=val)[2])
                R[p][(par, wo, wi)] = (s[0] - s0[0], s[1] - s0[1])
    print(p, f"RMSE v base {RMS[-1][1]:.2f} net {RMS[-1][2]:.2f} corr of errors {RMS[-1][3]:.3f} | {time.time() - T00:.0f}s", flush=True); load.cache_clear()
pickle.dump(dict(R=R, RMS=RMS), open(WORK + "/t1/g168_eval.pkl", "wb"))
obj = lambda par, wo, wi: np.array([0.35 * R[p][(par, wo, wi)][0] + 0.15 * R[p][(par, wo, wi)][1] for p in PANELS])
for sel, rd, nm in ((0, 1, "chosen on odd days, read on even days (the line)"), (1, 0, "chosen on even days, read on odd days (record)")):
    b = max(((wo, wi) for wo in WO for wi in WI), key=lambda c: obj(sel, *c).mean()); v = obj(rd, *b); dt = np.array([R[p][(rd, *b)][0] for p in PANELS]); dp = np.array([R[p][(rd, *b)][1] for p in PANELS])
    print(f"{nm}: w_out {b[0]} w_in {b[1]} | selection half {obj(sel, *b).mean():+.5f} | read half {v.mean():+.5f} (SE {v.std(ddof=1) / np.sqrt(len(v)):.5f}) | dS_state {dt.mean():+.5f} dS_phys {dp.mean():+.5f} | panels > 0: {(v > 0).sum()}/10 | " + " ".join(f"{x:+.4f}" for x in v), flush=True)
print("grid, both halves:"); [print(f"  w_out {wo} w_in {wi}: odd {obj(0, wo, wi).mean():+.5f} even {obj(1, wo, wi).mean():+.5f}") for wo in WO for wi in WI]
