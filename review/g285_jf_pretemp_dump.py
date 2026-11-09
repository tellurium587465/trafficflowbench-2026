"""G285 (11/05, ledger #234 revision, data before any temperature is chosen): the 316 January / February selector ongoing windows, G119's
SCG flow as G229's base column (capB0 s0-3 on the unmasked-history inputs probs_jfmh0__, the cone, the seen x0.85 under the "mask"
context), written out just before best_set (= where A applies its temperature), per (window, k) with the k weight (cap 3), plus the labels
and the scored cells. Identity: best_set + the far gate on the dumped probabilities reproduce G229's base column (g229_jf.csv).
Writes work/t2/g285_jf_pretemp.pkl. usage: TFB_M3VER=v3 py -B -X utf8 g285_jf_pretemp_dump.py"""
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
DUMP = {}; rows = []
for mo in g.MONTHS:
    for p in sorted({str(T.META["panel"][k]) for k in KS if MON[k] == mo}):
        P = g.load(p, "train"); key = (mo, p, "mask"); g.CTX.clear(); g.sim_blank.cache_clear(); g.CTX[key] = g.Ctx(key, P, np.where(P.blank, np.nan, P.V), P.V); c = g.CTX[key]
        for k in [k for k in KS if MON[k] == mo and str(T.META["panel"][k]) == p]:
            t = int(T.META["t"][k]); dom = int(P.times[t][8:10]) - 1; lab, el = T.LAB[k]; y, e = lab.astype(bool), el.astype(bool)
            lq = np.asarray(g.XN[g.OFFN[k]:g.OFFN[k + 1]][:, g.G.CI], float) == 1; p_ = sum(d[str(k)] for d in PR) / len(PR)
            cone = g.cone_parts(c, t, lq, P.length); ent = dict(y=y, e=e, month=mo, panel=p, day=dom, k={}); base = 0.0
            for kk, pr in g.m3v2.kdist(dom, cap=3).items():
                pc = p_ * g.cone_f(c, cone, t, kk); s, nd = g.seen(c, t, kk, -18, 12)
                pb = pc * np.where(g.G.dil(s, 1), 1.0, 0.85)[None, :] if nd else pc.copy()
                ent["k"][kk] = (pr, pb); base += pr * g.iou(g.far_gate(c, g.decode(pb), t, kk, lq), y, e, "queue_ongoing")
                assert np.array_equal(g.far_gate(c, g.decode(pb), t, kk, lq), g.ongoing_pred(c, pc, t, kk, True, lq))
            DUMP[int(k)] = ent; rows.append(dict(w=int(k), base=base))
        g.load.cache_clear()
pickle.dump(DUMP, open(g.W + "g285_jf_pretemp.pkl", "wb")); D = pd.DataFrame(rows).set_index("w"); R9 = pd.read_csv(g.W + "g229_jf.csv").set_index("w").base
print("windows dumped", len(DUMP), "| identity base against g229_jf.csv: max |d|", float(np.abs(D.base - R9.reindex(D.index)).max()), flush=True)
