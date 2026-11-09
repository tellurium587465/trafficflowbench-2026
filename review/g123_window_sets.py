"""G123 (10/30, ledger #77, strategy J17): is the M3-dev vs public-LB gap a window-set gap? The official Task 2 windows
exist on the training side only in 2030-06 (80); March's official futures are removed from the release. June: (a) the
official windows found in the evaluator (panel, T, condition), (b') the selector-like June windows not in (a), (b) all
selector-like June windows. Month-CV OOF q25_nomem s3/s4 (June left out), the G119 path with the mask visibility:
onset raw / E, ongoing raw / SG / SCG. Main: (b') - (a) for ongoing SCG and onset E.
usage: py g123_window_sets.py"""
import sys, time, glob, pickle, importlib.util
import numpy as np, pandas as pd
TM = time.perf_counter; T00 = TM(); R = r"C:/Users/user/tfb/"
spec = importlib.util.spec_from_file_location("g119", R + "review/g119_month_postproc.py"); g = importlib.util.module_from_spec(spec); spec.loader.exec_module(g)
T, G = g.T, g.G; MO = "2030-06"
OOF = [pickle.load(open(g.W + f"oof_q25_nomem_s{s}.pkl", "rb")) for s in (3, 4)]
idx = {}
for k in np.flatnonzero(g.MON == MO):
    idx.setdefault((str(T.META["panel"][k]), int(T.META["t"][k]), str(T.META["cond"][k])), []).append(int(k))
OFF = pd.concat([pd.read_csv(f) for f in glob.glob(R + "data/kaggle_public/task2/*/train/window_index.csv")])
off = set(); miss = 0
for p, d in OFF.groupby("panel"):
    tm = {str(x)[:16]: i for i, x in enumerate(g.load(p, "train").times)}
    for r in d.itertuples():
        ks = [k for k in idx.get((p, tm.get(r.forecast_origin[:16], -1), r.condition), []) if k in OOF[0] and k in OOF[1]]
        ks.sort(key=lambda k: not T.META["is_sel"][k])               # one evaluator row per window, the selector-like one first
        off.update(ks[:1]); miss += not ks
    g.load.cache_clear()
sel = {int(k) for k in np.flatnonzero(T.META["is_sel"] & (g.MON == MO)) if k in OOF[0] and k in OOF[1]}
print(f"official June windows {len(OFF)}: found {len(OFF) - miss} ({len(off)} evaluator rows, {len(off & sel)} selector-like) | selector-like {len(sel)}", flush=True)
rows = []
for p in sorted({str(T.META["panel"][k]) for k in off | sel}):
    P = g.load(p, "train"); key = (MO, p, "mask"); g.CTX.clear(); g.sim_blank.cache_clear()
    g.CTX[key] = g.Ctx(key, P, np.where(P.blank, np.nan, P.V), P.V); c = g.CTX[key]
    for k in sorted(k for k in off | sel if str(T.META["panel"][k]) == p):
        t = int(T.META["t"][k]); cond = str(T.META["cond"][k]); dom = int(P.times[t][8:10]) - 1; lab, el = T.LAB[k]
        Xw = np.asarray(g.XN[g.OFFN[k]:g.OFFN[k + 1]]); lq = np.asarray(Xw[:, G.CI], float) == 1; p_ = (OOF[0][k] + OOF[1][k]) / 2
        r = g.score_window(c, p_, t, cond, lab.astype(bool), el.astype(bool), lq, dom, P.length, np.zeros(P.n), 0.0)
        rows.append(dict(w=k, cond=cond, panel=p, day=dom, off=k in off, sel=k in sel, obs=float(el[:, 5].mean()), **r))
    g.load.cache_clear()
D = pd.DataFrame(rows); D.to_csv(g.W + "g123_window_sets.csv", index=False)
print(f"scored {len(D)} windows in {TM() - T00:.0f}s | label coverage at T+30: official {D[D.off].obs.mean():.3f}, others {D[~D.off].obs.mean():.3f}")


def mse(d, col, n_boot=2000, seed=0):
    cl = pd.factorize(d.panel + "_" + d.day.astype(str))[0]; M = np.bincount(cl, d[col].to_numpy()); N = np.bincount(cl)
    Rm = np.random.default_rng(seed).multinomial(len(M), np.full(len(M), 1.0 / len(M)), size=n_boot)
    return d[col].mean(), float(((Rm @ M) / (Rm @ N)).std())


for cond, cols in (("queue_onset", ("raw", "E")), ("queue_ongoing", ("raw", "SG", "SCG"))):
    d = D[D.cond == cond]; A_, B1, B = d[d.off], d[d.sel & ~d.off], d[d.sel]
    print(f"{cond}: (a) official {len(A_)} | (b') selector-like not official {len(B1)} | (b) selector-like {len(B)}")
    for col in cols:
        (ma, sa), (mb1, sb1), (mb, sb) = mse(A_, col), mse(B1, col), mse(B, col)
        print(f"  {col:4s} (a) {ma:.4f} ({sa:.4f}) | (b') {mb1:.4f} ({sb1:.4f}) | (b) {mb:.4f} | (b') - (a) {mb1 - ma:+.4f} (SE {np.hypot(sa, sb1):.4f}, "
              f"z {(mb1 - ma) / np.hypot(sa, sb1):+.2f}) | (b) - (a) {mb - ma:+.4f}")
