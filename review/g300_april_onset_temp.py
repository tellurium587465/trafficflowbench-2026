"""G300 (11/05, ledger #238, the third condition): do A17's April onset rows change with the onset temperature 1.3? As G289 (A16 = A17 on
onset: the saved blend probs_repro_finA16tcone.pkl + the evidence rule B = 0.25), then the temperature on the T+30 row and the full
best_set; counted against the T = 1 sets (which equal the submitted T+30 rows, G289's identity). usage: py -B -X utf8 g300_april_onset_temp.py"""
import sys, pickle
import numpy as np, pandas as pd
sys.path.insert(0, r"C:/Users/user/tfb_a8/src")
from cachelib import load, T2_PANELS, asof
from common import REL
from t2_model import best_set
W = "C:/Users/user/tfb/work/t2/"; lg = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
probs = pickle.load(open(W + "probs_repro_finA16tcone.pkl", "rb")); Q = pd.read_parquet(W + "queue_repro_finA17t.parquet"); QI = Q.groupby("window_id", sort=False).indices
nw = nc = add = rem = bad = n = 0
for p in T2_PANELS:
    P = load(p, "private"); ti = {s: i for i, s in enumerate(P.times)}
    for r in pd.read_csv(REL / "task2" / p / "private" / "window_index.csv").itertuples():
        if r.condition != "queue_onset":
            continue
        t = ti[r.forecast_origin]; pb = probs[r.window_id].copy(); Pt = asof(P, t); q = (Pt.V <= Pt.vcut[None, :]) & np.isfinite(Pt.V); seen = np.zeros(P.n, bool); nd = 0
        for back in (1, 2, 3):
            dd = t - back * 288
            if dd - 6 >= 0:
                seen |= q[dd - 6:dd + 12].any(axis=0); nd += 1
        if nd:
            ev = seen.copy()
            for o in (1, 2):
                ev[o:] |= seen[:-o]; ev[:-o] |= seen[o:]
            pb[-1] = pb[-1] * np.where(ev, 1.0, 0.25)
        s1 = best_set(pb.ravel(), np.ones(pb.size, bool)).reshape(pb.shape); x = pb.copy(); x[-1] = 1 / (1 + np.exp(-lg(pb[-1]) / 1.3)); s3 = best_set(x.ravel(), np.ones(x.size, bool)).reshape(x.shape)
        qs = Q.queue_pred.to_numpy()[QI[r.window_id]].reshape(pb.shape[0], P.n).astype(bool); bad += int(not np.array_equal(s1[-1], qs[-1])); n += 1
        d = int((s3[-1] != s1[-1]).sum()); nw += d > 0; nc += d; add += int((s3[-1] & ~s1[-1]).sum()); rem += int((s1[-1] & ~s3[-1]).sum())
print(f"April onset windows {n} | identity T 1 against A17's queue T+30: {n - bad} / {n} | T 1.3 changes {nw} windows, {nc} cells (added {add}, removed {rem})", flush=True)
