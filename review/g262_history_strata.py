"""G262 (11/04, ledger #224, report material, no decision): January / February 316 selector ongoing windows, IoU of G229's "trow"
flow (capB0 s0-3, G119's SCG flow with the row T as the last queue: the A-side post-processing of A11-A13; A13's own X1 is not computed
on these months) by thirds of (i) the number of missing cells in the 12 history rows T-12 .. T-1 under the masked context (the Task 1
blanks as the official windows) and (ii) the staleness of the last observation (rows since the latest finite cell in T-12 .. T-1, mean
over the links, 13 if none). Training split only. usage: TFB_M3VER=v3 py -B -X utf8 g262_history_strata.py"""
import os, sys, importlib.util
os.environ.setdefault("TFB_M3VER", "v3")
import numpy as np, pandas as pd
R = r"C:/Users/user/tfb/"; a_ = sys.argv; sys.argv = ["g119", "run"]
spec = importlib.util.spec_from_file_location("g119", R + "review/g119_month_postproc.py"); g = importlib.util.module_from_spec(spec); spec.loader.exec_module(g); sys.argv = a_
T, MON = g.T, g.MON; J = pd.read_csv(g.W + "g229_jf.csv").set_index("w"); rows = []
KS = [k for k in np.flatnonzero(T.META["is_sel"] & np.isin(MON, g.MONTHS)) if str(T.META["cond"][k]) == "queue_ongoing"]
for mo in g.MONTHS:
    for p in sorted({str(T.META["panel"][k]) for k in KS if MON[k] == mo}):
        P = g.load(p, "train"); key = (mo, p, "mask"); g.CTX.clear(); g.sim_blank.cache_clear(); g.CTX[key] = g.Ctx(key, P, np.where(P.blank, np.nan, P.V), P.V); c = g.CTX[key]
        for k in [k for k in KS if MON[k] == mo and str(T.META["panel"][k]) == p]:
            t = int(T.META["t"][k]); ob = np.asarray(c.obs[t - 12:t], bool); nmiss = int((~ob).sum())
            last = np.array([12 - np.flatnonzero(ob[:, j]).max() if ob[:, j].any() else 13 for j in range(ob.shape[1])], float)
            rows.append(dict(w=int(k), month=mo, panel=p, nmiss=nmiss, frac_miss=nmiss / ob.size, stale=float(last.mean()), iou=float(J.trow[int(k)])))
        g.load.cache_clear()
D = pd.DataFrame(rows); D.to_csv(g.W + "g262_jf.csv", index=False)
for col in ("frac_miss", "stale"):
    D["q"] = pd.qcut(D[col].rank(method="first"), 3, labels=["low", "mid", "high"])
    s = D.groupby("q", observed=True).agg(n=("iou", "size"), lo=(col, "min"), hi=(col, "max"), iou=("iou", "mean"), sd=("iou", "std"))
    s["se"] = s.sd / np.sqrt(s.n); print(f"--- by {col} (thirds)\n{s.round(4).to_string()}", flush=True)
print("corr(frac_miss, iou)", round(D.frac_miss.corr(D.iou), 3), "| corr(stale, iou)", round(D.stale.corr(D.iou), 3), "| corr(frac_miss, stale)", round(D.frac_miss.corr(D.stale), 3))
