"""G306 (11/05, ledger #244 V1, ongoing): does the M3-type history mask explain why u2d reads negative on M3 and positive on January / February?
The 316 January / February selector ongoing windows (G220's KS). (1) The raw history R (t2_raw_hist.py, rows T-12 .. T) rebuilt in a train-
layout directory with TFB_T2_MASKHIST=0 (identity: equals nn_data_afdem/R.npy on these windows) and =1 (M3-type mask on T-12 .. T-1).
(2) u2d s0, s1 (fold 2030-12, G193's network on the CPU) on both (identity: mh0 equals oof_u2d_s*). (3) G119's capB0 flow as G285 / G229
(cone, seen x0.85, best_set, far gate under the "mask" context, k expectation cap 3) on capB0 jfmh0 / jfmh1 with and without u2d at 0.2
(0.8 capB0 + 0.2 u2d, as A mixes it); the effect under mh0 and mh1 and the paired mh1 - mh0, cluster bootstrap (month x panel x day).
usage: TFB_M3VER=v3 py -B -X utf8 g306_jf_mask_u2d.py"""
import os, sys, pickle, subprocess, importlib.util
os.environ.setdefault("TFB_M3VER", "v3")
import numpy as np, pandas as pd, torch
R = r"C:/Users/user/tfb/"; NN = "C:/tfb_data/tfb_nn/"; W = R + "work/t2/"; d = NN + "nn_data_afdem/"
date = np.load(d + "date.npy", allow_pickle=True); cond = np.load(d + "cond.npy", allow_pickle=True); isel = np.load(d + "is_sel.npy")
pan = np.load(d + "panel.npy", allow_pickle=True); t_ = np.load(d + "t.npy"); off = np.load(d + "off.npy"); mon = np.array([str(x)[:7] for x in date])
KS = np.flatnonzero(isel & np.isin(mon, ("2031-01", "2031-02")) & (cond == "queue_ongoing")); n_ = np.diff(off)[KS]; OF = np.concatenate([[0], np.cumsum(n_)])
for mh in "01":
    o = NN + f"nn_jfogR{mh}/"; os.makedirs(o, exist_ok=True); np.save(o + "panel.npy", pan[KS]); np.save(o + "t.npy", t_[KS]); np.save(o + "off.npy", OF)
    if not os.path.exists(o + "R.npy"):
        subprocess.run([sys.executable, "-B", R + "src/t2_raw_hist.py", f"nn_jfogR{mh}"], check=True, cwd=R + "src", env=dict(os.environ, TFB_T2_MASKHIST=mh))
RA = np.load(d + "R.npy", mmap_mode="r"); R0 = np.load(NN + "nn_jfogR0/R.npy"); R1 = np.load(NN + "nn_jfogR1/R.npy")
dm = max(float(np.abs(np.asarray(RA[off[k]:off[k + 1]], float) - R0[OF[i]:OF[i + 1]].astype(float)).max()) for i, k in enumerate(KS))
print(f"windows {len(KS)} | identity R mh0 against nn_data_afdem/R.npy: max |d| {dm:.2e} | mh1 vs mh0 cells differing {int((R1 != R0).sum())} of {R0.size}", flush=True)
src = open(R + "review/g193_u2d_infer.py", encoding="utf-8").read(); g = {}
exec(compile(src.split("\nd = NN + \"nn_test_m3v3/\"")[0].replace("SEED = int(sys.argv[1]);", "SEED = 0;"), "g193", "exec"), g)
U = {}
for s_ in (0, 1):
    g["net"].load_state_dict(torch.load(f"C:/tfb_data/kaggle_out/u2d/u2d_s{s_}_fold.pt", map_location="cpu")); g["net"].eval()
    for mh, RR in (("0", R0), ("1", R1)):
        r = g["run"](RR, OF, range(len(KS)), np.zeros(len(KS), bool)); U[(s_, mh)] = {int(KS[i]): v for i, v in r.items()}
OO = [pickle.load(open(W + f"oof_u2d_s{s_}.pkl", "rb")) for s_ in (0, 1)]; kk0 = next(iter(OO[0])); cv = (lambda k: k) if isinstance(kk0, (int, np.integer)) else str
print("identity u2d mh0 against oof_u2d: max |d|", max(float(np.abs(U[(s_, "0")][int(k)] - OO[s_][cv(int(k))]).max()) for s_ in (0, 1) for k in KS), flush=True)


def module(nm, path, argv):
    a = sys.argv; sys.argv = argv; spec = importlib.util.spec_from_file_location(nm, path); mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); sys.argv = a
    return mod


h = module("g119", R + "review/g119_month_postproc.py", ["g119", "run"]); T, MON = h.T, h.MON
CB = {mh: [pickle.load(open(W + f"probs_jfmh{mh}__capB0_s{s_}_fold2030-12.pkl", "rb")) for s_ in range(4)] for mh in "01"}; rows = []
for mo in h.MONTHS:
    for p in sorted({str(T.META["panel"][k]) for k in KS if MON[k] == mo}):
        P = h.load(p, "train"); key = (mo, p, "mask"); h.CTX.clear(); h.sim_blank.cache_clear(); h.CTX[key] = h.Ctx(key, P, np.where(P.blank, np.nan, P.V), P.V); c = h.CTX[key]
        for k in [k for k in KS if MON[k] == mo and str(T.META["panel"][k]) == p]:
            t = int(T.META["t"][k]); dom = int(P.times[t][8:10]) - 1; lab, el = T.LAB[k]; y, e = lab.astype(bool), el.astype(bool)
            lq = np.asarray(h.XN[h.OFFN[k]:h.OFFN[k + 1]][:, h.G.CI], float) == 1; cone = h.cone_parts(c, t, lq, P.length); r = dict(w=int(k), month=mo, panel=p, day=dom)
            for mh in "01":
                cb = sum(d_[str(k)] for d_ in CB[mh]) / 4; uu = (U[(0, mh)][int(k)] + U[(1, mh)][int(k)]) / 2
                for nm, pp in ((f"c{mh}", cb), (f"c{mh}u", 0.8 * cb + 0.2 * uu)):
                    r[nm] = sum(pr * h.iou(h.ongoing_pred(c, pp * h.cone_f(c, cone, t, kk), t, kk, True, lq), y, e, "queue_ongoing") for kk, pr in h.m3v2.kdist(dom, cap=3).items())
            rows.append(r)
        h.load.cache_clear()
D = pd.DataFrame(rows); D.to_csv(W + "g306_jf.csv", index=False); R9 = pd.read_csv(W + "g229_jf.csv").set_index("w").base
print("identity c0 against g229 base:", float(np.abs(D.set_index("w").c0 - R9.reindex(D.w).to_numpy()).max()), flush=True)
cl = pd.factorize(D.month + "_" + D.panel + "_" + D.day.astype(str))[0]; Rm = np.random.default_rng(0).multinomial(cl.max() + 1, np.full(cl.max() + 1, 1 / (cl.max() + 1)), size=2000)
bse = lambda d_: float(((Rm @ np.bincount(cl, d_)) / (Rm @ np.bincount(cl))).std())
e0 = (D.c0u - D.c0).to_numpy(); e1 = (D.c1u - D.c1).to_numpy(); dd = e1 - e0
print(f"capB0 level: mh0 {D.c0.mean():.4f}, mh1 {D.c1.mean():.4f} (mask cost {(D.c1 - D.c0).mean():+.4f})", flush=True)
print(f"u2d effect (0.8 capB0 + 0.2 u2d - capB0): mh0 (official-like) {e0.mean():+.5f} (SE {bse(e0):.5f}) | mh1 (M3-like) {e1.mean():+.5f} (SE {bse(e1):.5f}) | mh1 - mh0 {dd.mean():+.5f} (SE {bse(dd):.5f}, z {dd.mean() / max(bse(dd), 1e-12):+.2f}) | windows where the effect moves {int((np.abs(dd) > 1e-12).sum())}", flush=True)
