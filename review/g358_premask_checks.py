"""G358 (11/05, ledger #284 b-e, measurement only). (a') offsets of V11P / V6P from g354's csv (combined gap). (b) the island-width formula:
for the 40 official June ongoing windows and k = 1 .. 6, the origin T-k's visible coverage (Task 1 blank hidden) against the pre-mask
coverage c (finite & eligible) of rows T-k-12 .. T-k-1; predicted "visible < 0.7" iff k > 12 (1 - 0.7 / c) / r, r = the blank share
of the finite & eligible cells in the rows that leave the island (T-24 .. T-13); agreement table. (c) the T row's P.blank share on the 80
official June windows (per cell) against the month's blank share. (d) the training split: count of (P.E & ~isfinite(P.V)). (e) the
validation (March) official windows: eligible & missing cells in their history rows T-12 .. T-1. usage: py -B -X utf8 g358_premask_checks.py"""
import sys, pickle, glob
import numpy as np, pandas as pd
sys.path.insert(0, r"C:/Users/user/tfb/src")
from cachelib import load, T2_PANELS
from common import REL
W = "C:/Users/user/tfb/work/t2/"; A = pickle.load(open(W + "g342_jun80.pkl", "rb"))
D = pd.read_csv(W + "g354_selector_gap72_tau0.9_cov0.7.csv") if glob.glob(W + "g354_selector_gap72_tau0.9_cov0.7.csv") else pd.read_csv(sorted(glob.glob(W + "g354_selector_gap72_tau0.9*.csv"))[-1])
for v in ("V11P", "V6P"):
    d = D[(D["var"] == v) & (D.cond == "queue_ongoing")]
    if len(d):
        print(f"(a') {v} ongoing offsets (rows after the first qualifying origin): {d.off_rows.value_counts().sort_index().to_dict()}", flush=True)
rows = []; tb = []; dE = 0; eM = []
for p in T2_PANELS:
    P = load(p, "train"); mon = np.array([s[:7] for s in P.times]); fe = np.isfinite(P.V) & P.E; vis = fe & ~P.blank
    dE += int((P.E & ~np.isfinite(P.V)).sum()); jb = float(P.blank[mon == "2030-06"].mean())
    for r in [x for x in A["rows"] if x["panel"] == p]:
        t = r["t"]; tb.append(dict(blankT=float(P.blank[t].mean()), month_blank=jb))
        if r["cond"] != "queue_ongoing":
            continue
        out_rows = fe[t - 24:t - 12]; rr = float((P.blank[t - 24:t - 12] & out_rows).sum() / max(out_rows.sum(), 1))
        for k in range(1, 7):
            a, b = t - k - 12, t - k; c = float(fe[a:b].mean()); v = float(vis[a:b].mean())
            pred = k > 12 * (1 - 0.7 / c) / max(rr, 1e-9) if c > 0.7 else True
            rows.append(dict(k=k, c=c, v=v, r=rr, pred=pred, obs=v < 0.7))
    load.cache_clear()
R = pd.DataFrame(rows); print(f"(b) origins T-1 .. T-6 of 40 ongoing windows: predicted / observed 'visible < 0.7':\n{pd.crosstab(R.pred, R.obs).to_string()}\n    agreement {float((R.pred == R.obs).mean()):.3f}; by k: " + ", ".join(f"k{k} obs {R[R.k == k].obs.mean():.2f} pred {R[R.k == k].pred.mean():.2f}" for k in range(1, 7)), flush=True)
B = pd.DataFrame(tb); print(f"(c) row T P.blank share on the 80 official June windows: mean {B.blankT.mean():.3f} (windows with any blank at T {(B.blankT > 0).sum()}), June blank share {B.month_blank.mean():.3f}", flush=True)
print(f"(d) training split: cells with E & not finite V = {dE}", flush=True)
for p in T2_PANELS:
    P = load(p, "validation"); ti = {s: i for i, s in enumerate(P.times)}
    for r in pd.read_csv(REL / "task2" / p / "validation" / "window_index.csv").itertuples():
        t = ti[r.forecast_origin]; eM.append(int((P.E[t - 12:t] & ~np.isfinite(P.V[t - 12:t])).sum()))
    load.cache_clear()
eM = np.array(eM); print(f"(e) March official windows: history cells eligible & missing: total {eM.sum()}, windows with any {(eM > 0).sum()} of {len(eM)}", flush=True)
