"""G229 (11/03, ledger #197): the row at T in the post-processing only. The far gate and the cone start from the last queue lq (each
link's latest finite observation, T-60 .. T-5, the feature column G.CI); here a link observed at the row T takes its queue state there
instead (lq2 = where(obs[T], q[T], lq)); the networks' inputs are unchanged. Instrument (#188): the 316 January / February selector
ongoing windows, probabilities = capB0 seeds 0-3 on the unmasked-history inputs (probs_jfmh0__*), G119's SCG flow (seen x0.85,
best_set, far gate, cone) under its "mask" context (the Task 1 blank hidden except the window's T-12 .. T-1, as the official windows;
the row T is visible on 48-67 % of the links). k expectation (cap 3). Paired lq2 - lq, cluster bootstrap (month x panel x day).
usage: TFB_M3VER=v3 py -B -X utf8 g229_trow_postproc.py"""
import os, sys, pickle, importlib.util
os.environ.setdefault("TFB_M3VER", "v3")
import numpy as np, pandas as pd
R = r"C:/Users/user/tfb/"


def module(nm, path, argv):
    a = sys.argv; sys.argv = argv; spec = importlib.util.spec_from_file_location(nm, path); mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); sys.argv = a
    return mod


g = module("g119", R + "review/g119_month_postproc.py", ["g119", "run"]); T, MON = g.T, g.MON
KS = [k for k in np.flatnonzero(T.META["is_sel"] & np.isin(MON, g.MONTHS)) if str(T.META["cond"][k]) == "queue_ongoing"]
PR = [pickle.load(open(g.W + f"probs_jfmh0__capB0_s{s}_fold2030-12.pkl", "rb")) for s in range(4)]
rows = []
for mo in g.MONTHS:
    for p in sorted({str(T.META["panel"][k]) for k in KS if MON[k] == mo}):
        P = g.load(p, "train"); key = (mo, p, "mask"); g.CTX.clear(); g.sim_blank.cache_clear(); g.CTX[key] = g.Ctx(key, P, np.where(P.blank, np.nan, P.V), P.V); c = g.CTX[key]
        for k in [k for k in KS if MON[k] == mo and str(T.META["panel"][k]) == p]:
            t = int(T.META["t"][k]); dom = int(P.times[t][8:10]) - 1; lab, el = T.LAB[k]; y, e = lab.astype(bool), el.astype(bool)
            lq = np.asarray(g.XN[g.OFFN[k]:g.OFFN[k + 1]][:, g.G.CI], float) == 1; ot = c.obs[t].astype(bool); lq2 = np.where(ot, c.q[t].astype(bool), lq)
            p_ = sum(d[str(k)] for d in PR) / len(PR); r = dict(w=int(k), month=mo, panel=p, day=dom, obsT=float(ot.mean()), lq_changed=int((lq2 != lq).sum()))
            for nm, L_ in (("base", lq), ("trow", lq2)):
                cone = g.cone_parts(c, t, L_, P.length)
                r[nm] = sum(pr * g.iou(g.ongoing_pred(c, p_ * g.cone_f(c, cone, t, kk), t, kk, True, L_), y, e, "queue_ongoing") for kk, pr in g.m3v2.kdist(dom, cap=3).items())
            rows.append(r)
        g.load.cache_clear()
D = pd.DataFrame(rows); D.to_csv(g.W + "g229_jf.csv", index=False)


def boot(d, cl, n_boot=2000, seed=0):
    rng = np.random.default_rng(seed); ids = pd.factorize(cl)[0]; M = np.bincount(ids, d); N = np.bincount(ids)
    Rm = rng.multinomial(len(M), np.full(len(M), 1.0 / len(M)), size=n_boot); return float(((Rm @ M) / (Rm @ N)).std())


d = (D.trow - D.base).to_numpy(); cl = (D.month + "_" + D.panel + "_" + D.day.astype(str)).to_numpy(); se = boot(d, cl); odd = (D.day % 2 == 0).to_numpy()
print(f"{len(D)} windows (row T observed on {D.obsT.mean():.2f} of the links; lq changed on {int((D.lq_changed > 0).sum())} windows) | base {D.base.mean():.4f}")
print(f"trow - base: merged {d.mean():+.4f} (SE {se:.4f}, z {d.mean() / max(se, 1e-12):+.2f}, z/1.25 {d.mean() / max(se, 1e-12) / 1.25:+.2f}) | "
      + " | ".join(f"{mo} {d[(D.month == mo).to_numpy()].mean():+.4f}" for mo in g.MONTHS) + f" | odd / even days {d[odd].mean():+.4f} / {d[~odd].mean():+.4f} | windows changed {int((np.abs(d) > 1e-12).sum())}")
