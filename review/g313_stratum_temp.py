"""G313 (11/05, ledger #249): the ongoing temperature of the young or small boxes only. Stratum (L2's definition rebuilt from the raw history R,
rows T-12 .. T-1): the queued links of the last visible row total <= 3, or the box has been visible for <= 3 consecutive rows; 27.0 % of M3 dev
ongoing (L2: 24 %), 28.5 % of the January / February selector ongoing windows. Within the stratum's windows, #234's method on G284 / G285's
dumps: cell log loss on the #234 grid, T* by argmin, window bootstrap 2000 (M3 jointly for A17's mix and capB0, January / February apart),
delta = T*_JF(capB0) - T*_M3(capB0); bracket [T*_M3(A), T*_M3(A) + delta]: both ends exclude 0.8 (today's T) on the same side -> T_S = the end
nearer 0.8 (0.05 steps). Then the window IoU of T_S on the stratum (0.8 elsewhere) against 0.8 everywhere: M3 dev A17 (G287's decode,
m3v2.summary) and January / February capB0 (G287's flow, cluster bootstrap). usage: TFB_M3VER=v3 py -B -X utf8 g313_stratum_temp.py"""
import os, sys, pickle, importlib.util
os.environ.setdefault("TFB_M3VER", "v3")
import numpy as np, pandas as pd
R = r"C:/Users/user/tfb/"; sys.path.insert(0, R + "src")
import t2_m3_gate as G, m3v2
from t2_model import best_set
W = R + "work/t2/"; TS = np.array([0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.8, 2.0])
lg = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6))); tmp = lambda p, T: 1 / (1 + np.exp(-lg(p) / T))
bs = lambda pb: best_set(pb.ravel(), np.ones(pb.size, bool)).reshape(pb.shape)
ST = pickle.load(open(W + "g313_strata.pkl", "rb")); D4 = pickle.load(open(W + "g284_og_pretemp_dev.pkl", "rb")); D5 = pickle.load(open(W + "g285_jf_pretemp.pkl", "rb"))
WM = sorted(w for w in D4 if ST["M"][w][0]); KJ = sorted(k for k in D5 if ST["J"][k][0])


def tab(E):
    S = np.zeros((len(E), len(TS))); N = np.zeros(len(E))
    for i, (y, msk, ks) in enumerate(E):
        yy = y[msk].astype(float)
        for pr, p in ks:
            z = lg(p[msk].astype(float)); N[i] += pr * msk.sum()
            for j, T in enumerate(TS):
                q = np.clip(1 / (1 + np.exp(-z / T)), 1e-7, 1 - 1e-7); S[i, j] += pr * float(-(yy * np.log(q) + (1 - yy) * np.log(1 - q)).sum())
    return S, N


SA, NA = tab([(D4[w]["y"].T.astype(bool), D4[w]["m"].T.astype(bool), [(v[0], v[1]) for v in D4[w]["k"].values()]) for w in WM])
SC, NC = tab([(D4[w]["y"].T.astype(bool), D4[w]["m"].T.astype(bool), [(v[0], v[4]) for v in D4[w]["k"].values()]) for w in WM])
SJ, NJ = tab([(D5[k]["y"], D5[k]["e"], list(D5[k]["k"].values())) for k in KJ])
FR = pd.DataFrame([dict(w=w, cond=G.META[w]["cond"], panel=G.META[w]["panel"], day=G.META[w]["day"], hour=(G.META[w]["t"] % 288) // 12) for w in WM]); WT = m3v2.weights(FR, "private")[0]; oJ = np.ones(len(SJ))
ts = lambda S, N, wt, ix: TS[((wt[ix][:, None] * S[ix]).sum(axis=0) / (wt[ix] * N[ix]).sum()).argmin()]
for nm, S, N, wt in (("M3 A17 stratum", SA, NA, WT), ("M3 capB0 stratum", SC, NC, WT), ("JF capB0 stratum", SJ, NJ, oJ)):
    c = (wt[:, None] * S).sum(axis=0) / (wt * N).sum(); print(f"{nm} ({len(S)} windows): T* {TS[c.argmin()]} | log loss " + " ".join(f"{T}:{v:.5f}" for T, v in zip(TS, c)), flush=True)
rg = np.random.default_rng(0); tA, tC, tJ = [], [], []
for _ in range(2000):
    ix = rg.integers(0, len(SA), len(SA)); jx = rg.integers(0, len(SJ), len(SJ)); tA.append(ts(SA, NA, WT, ix)); tC.append(ts(SC, NC, WT, ix)); tJ.append(ts(SJ, NJ, oJ, jx))
tA, tC, tJ = map(np.array, (tA, tC, tJ)); dl = tJ - tC; q = lambda v: (float(np.quantile(v, 0.05)), float(np.quantile(v, 0.95)))
TA0 = ts(SA, NA, WT, np.arange(len(SA))); DL0 = ts(SJ, NJ, oJ, np.arange(len(SJ))) - ts(SC, NC, WT, np.arange(len(SC))); iA, iAd = q(tA), q(tA + dl)
side = lambda i: "above" if i[0] > 0.8 else ("below" if i[1] < 0.8 else "contains")
ok = side(iA) == side(iAd) and side(iA) != "contains"; TSel = round(min((TA0, TA0 + DL0), key=lambda v: abs(v - 0.8)) / 0.05) * 0.05 if ok else 0.8
print(f"T*_M3(A) {TA0} 90% [{iA[0]}, {iA[1]}] ({side(iA)} 0.8) | delta {DL0:+.2f} 90% [{q(dl)[0]:+.2f}, {q(dl)[1]:+.2f}] | T*_A + delta {TA0 + DL0:.2f} 90% [{iAd[0]:.2f}, {iAd[1]:.2f}] ({side(iAd)} 0.8) | bracket rule: {ok} -> T_S {TSel:.2f}", flush=True)
if not ok:
    sys.exit()


def module(nm, path, argv):
    a = sys.argv; sys.argv = argv; spec = importlib.util.spec_from_file_location(nm, path); mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); sys.argv = a
    return mod


g132 = module("g132", R + "review/g132_p3_grid.py", ["g132", "run"]); rows = []
for w, v in D4.items():
    m = G.META[w]; Ts = TSel if ST["M"][w][0] else 0.8
    rows.append(dict(w=w, cond=m["cond"], panel=m["panel"], day=m["day"], hour=(m["t"] % 288) // 12, t08=sum(pr * G.iou(g132.gate(bs(tmp(p, 0.8)), ex, nr), m) for pr, p, ex, nr, _ in v["k"].values()),
                     ts=sum(pr * G.iou(g132.gate(bs(tmp(p, Ts)), ex, nr), m) for pr, p, ex, nr, _ in v["k"].values())))
DM = pd.DataFrame(rows); sm = m3v2.summary(DM, "t08", "ts", "private"); R7 = pd.read_csv(W + "g287_m3.csv").set_index("w")["t0.8"]
print(f"identity t08 against g287: {float(np.abs(DM.set_index('w').t08 - R7.reindex(DM.w).to_numpy()).max()):.1e} | M3 dev A17, T_S on the stratum - 0.8: {sm['t2_weighted']:+.5f} (SE {sm['se']:.5f}, z {sm['t2_weighted'] / max(sm['se'], 1e-12):+.2f})", flush=True)
h = module("g119", R + "review/g119_month_postproc.py", ["g119", "run"]); T_, MON = h.T, h.MON
PR = [pickle.load(open(h.W + f"probs_jfmh0__capB0_s{s}_fold2030-12.pkl", "rb")) for s in range(4)]; rows = []
for mo in h.MONTHS:
    for p in sorted({str(T_.META["panel"][k]) for k in D5 if D5[k]["month"] == mo}):
        P = h.load(p, "train"); key = (mo, p, "mask"); h.CTX.clear(); h.sim_blank.cache_clear(); h.CTX[key] = h.Ctx(key, P, np.where(P.blank, np.nan, P.V), P.V); c = h.CTX[key]
        for k in [k for k in D5 if D5[k]["month"] == mo and D5[k]["panel"] == p]:
            t = int(T_.META["t"][k]); dom = D5[k]["day"]; y, e = D5[k]["y"], D5[k]["e"]; lq = np.asarray(h.XN[h.OFFN[k]:h.OFFN[k + 1]][:, h.G.CI], float) == 1; Ts = TSel if ST["J"][k][0] else 0.8
            r = dict(w=k, month=mo, panel=p, day=dom, t08=0.0, ts=0.0)
            for kk, (pr, pb) in D5[k]["k"].items():
                r["t08"] += pr * h.iou(h.far_gate(c, h.decode(tmp(pb, 0.8)), t, kk, lq), y, e, "queue_ongoing"); r["ts"] += pr * h.iou(h.far_gate(c, h.decode(tmp(pb, Ts)), t, kk, lq), y, e, "queue_ongoing")
            rows.append(r)
        h.load.cache_clear()
DJ = pd.DataFrame(rows); cl = pd.factorize(DJ.month + "_" + DJ.panel + "_" + DJ.day.astype(str))[0]; Rm = rg.multinomial(cl.max() + 1, np.full(cl.max() + 1, 1 / (cl.max() + 1)), size=2000); d = (DJ.ts - DJ.t08).to_numpy()
se = float(((Rm @ np.bincount(cl, d)) / (Rm @ np.bincount(cl))).std()); R8 = pd.read_csv(W + "g287_jf.csv").set_index("w")["t0.8"]
print(f"identity JF t08 against g287: {float(np.abs(DJ.set_index('w').t08 - R8.reindex(DJ.w).to_numpy()).max()):.1e} | JF capB0, T_S on the stratum - 0.8: {d.mean():+.5f} (SE {se:.5f}, z {d.mean() / max(se, 1e-12):+.2f})", flush=True)
w1, w2 = 1 / max(sm["se"], 1e-9) ** 2, 1 / max(se, 1e-9) ** 2; mg = (w1 * sm["t2_weighted"] + w2 * d.mean()) / (w1 + w2); smg = (w1 + w2) ** -0.5
print(f"merged {mg:+.5f} (SE {smg:.5f}, z {mg / smg:+.2f}) | line (both > 0, merged z >= +1): {sm['t2_weighted'] > 0 and d.mean() > 0 and mg / smg >= 1}", flush=True)
