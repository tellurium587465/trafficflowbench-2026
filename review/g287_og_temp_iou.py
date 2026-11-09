"""G287 (11/05, ledger #234, the second condition of the line): the window IoU at the temperatures of the grid against T = 1.2.
M3 dev: G284's dump (A16's March ongoing mix), best_set then G132's far gate with the dumped (ex, near), G.iou, k-weighted; m3v2.summary
(private weights, cluster bootstrap) of T - 1.2. Jan / Feb: G285's flow re-run (capB0, the "mask" context) with the temperature before
best_set and G119's far gate; cluster bootstrap (month x panel x day) of T - 1.2. Identity: T = 1.2 on M3 reproduces G276's a12 column.
usage: TFB_M3VER=v3 py -B -X utf8 g287_og_temp_iou.py"""
import os, sys, pickle, importlib.util
os.environ.setdefault("TFB_M3VER", "v3")
import numpy as np, pandas as pd
R = r"C:/Users/user/tfb/"; sys.path.insert(0, R + "src")
import t2_m3_gate as G, m3v2
from t2_model import best_set
W = R + "work/t2/"; TS = (0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.6, 2.0)
lg = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6))); tmp = lambda p, T: 1 / (1 + np.exp(-lg(p) / T))
bs = lambda pb: best_set(pb.ravel(), np.ones(pb.size, bool)).reshape(pb.shape)


def module(nm, path, argv):
    a = sys.argv; sys.argv = argv; spec = importlib.util.spec_from_file_location(nm, path); mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); sys.argv = a
    return mod


g132 = module("g132", R + "review/g132_p3_grid.py", ["g132", "run"])
D4 = pickle.load(open(W + "g284_og_pretemp_dev.pkl", "rb")); rows = []
for w, v in D4.items():
    m = G.META[w]; r = dict(w=w, cond=m["cond"], panel=m["panel"], day=m["day"], hour=(m["t"] % 288) // 12)
    for T in TS:
        r[f"t{T}"] = sum(pr * G.iou(g132.gate(bs(tmp(p, T)), ex, nr), m) for pr, p, ex, nr, _ in v["k"].values())
    rows.append(r)
DM = pd.DataFrame(rows); R6 = pd.read_csv(W + "g276_dev.csv").set_index("w").a12
print("M3 dev", len(DM), "| identity T 1.2 against g276 a12: max |d|", float(np.abs(DM.set_index("w")["t1.2"] - R6.reindex(DM.w).to_numpy()).max()), flush=True)
for T in TS:
    if T != 1.2:
        s = m3v2.summary(DM, "t1.2", f"t{T}", "private"); print(f"  M3 A  T {T} - 1.2: {s['t2_weighted']:+.5f} (SE {s['se']:.5f}, z {s['t2_weighted'] / max(s['se'], 1e-12):+.2f})", flush=True)
g = module("g119", R + "review/g119_month_postproc.py", ["g119", "run"]); T_, MON = g.T, g.MON
KS = [k for k in np.flatnonzero(T_.META["is_sel"] & np.isin(MON, g.MONTHS)) if str(T_.META["cond"][k]) == "queue_ongoing"]
PR = [pickle.load(open(g.W + f"probs_jfmh0__capB0_s{s}_fold2030-12.pkl", "rb")) for s in range(4)]; rows = []
for mo in g.MONTHS:
    for p in sorted({str(T_.META["panel"][k]) for k in KS if MON[k] == mo}):
        P = g.load(p, "train"); key = (mo, p, "mask"); g.CTX.clear(); g.sim_blank.cache_clear(); g.CTX[key] = g.Ctx(key, P, np.where(P.blank, np.nan, P.V), P.V); c = g.CTX[key]
        for k in [k for k in KS if MON[k] == mo and str(T_.META["panel"][k]) == p]:
            t = int(T_.META["t"][k]); dom = int(P.times[t][8:10]) - 1; lab, el = T_.LAB[k]; y, e = lab.astype(bool), el.astype(bool)
            lq = np.asarray(g.XN[g.OFFN[k]:g.OFFN[k + 1]][:, g.G.CI], float) == 1; p_ = sum(d[str(k)] for d in PR) / len(PR); cone = g.cone_parts(c, t, lq, P.length)
            r = dict(w=int(k), month=mo, panel=p, day=dom, **{f"t{T}": 0.0 for T in TS})
            for kk, pr in g.m3v2.kdist(dom, cap=3).items():
                pc = p_ * g.cone_f(c, cone, t, kk); s_, nd = g.seen(c, t, kk, -18, 12); pb = pc * np.where(g.G.dil(s_, 1), 1.0, 0.85)[None, :] if nd else pc.copy()
                for T in TS:
                    r[f"t{T}"] += pr * g.iou(g.far_gate(c, g.decode(tmp(pb, T) if T != 1.0 else pb), t, kk, lq), y, e, "queue_ongoing")
            rows.append(r)
        g.load.cache_clear()
DJ = pd.DataFrame(rows); DM.to_csv(W + "g287_m3.csv", index=False); DJ.to_csv(W + "g287_jf.csv", index=False)
R9 = pd.read_csv(g.W + "g229_jf.csv").set_index("w").base; print("Jan / Feb", len(DJ), "| identity T 1.0 against g229 base: max |d|", float(np.abs(DJ.set_index("w")["t1.0"] - R9.reindex(DJ.w).to_numpy()).max()), flush=True)
cl = pd.factorize(DJ.month + "_" + DJ.panel + "_" + DJ.day.astype(str))[0]; rng = np.random.default_rng(0); Rm = rng.multinomial(cl.max() + 1, np.full(cl.max() + 1, 1 / (cl.max() + 1)), size=2000)
for T in TS:
    if T != 1.2:
        d = (DJ[f"t{T}"] - DJ["t1.2"]).to_numpy(); M = np.bincount(cl, d); N = np.bincount(cl); se = float(((Rm @ M) / (Rm @ N)).std())
        print(f"  JF capB0 T {T} - 1.2: {d.mean():+.5f} (SE {se:.5f}, z {d.mean() / max(se, 1e-12):+.2f})", flush=True)
