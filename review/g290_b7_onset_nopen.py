"""G290 (11/05, ledger #236 stage 0 (a)): B7's April onset rows with the evidence penalty 0.25 (as built) against 1.0 (no penalty).
B7's saved blend (work/t2/probs_repro_finB7tcone.pkl, the input of t2_onset_evidence.py), the evidence rule as G289 / t2_onset_evidence
(links +-2 without queue in T-30 .. T+60 min on up to 3 earlier days, x B at T+30, SKIP 0), best_set. Identity: B = 0.25 reproduces the
T+30 row of B7's submitted queue on every April onset window. Counted: windows and T+30 cells whose set changes with B = 1.0.
Writes work/t2/g290_april_b7s.pkl (the B = 1.0 sets). usage: py -B -X utf8 g290_b7_onset_nopen.py"""
import sys, pickle
import numpy as np, pandas as pd
sys.path.insert(0, r"C:/Users/user/tfb_a8/src")
from cachelib import load, T2_PANELS, asof
from common import REL
from t2_model import best_set
W = "C:/Users/user/tfb/work/t2/"
probs = pickle.load(open(W + "probs_repro_finB7tcone.pkl", "rb")); Q = pd.read_parquet(W + "queue_repro_finB7t.parquet"); QI = Q.groupby("window_id", sort=False).indices
out = {}; bad = nw = nc = 0
for p in T2_PANELS:
    P = load(p, "private"); ti = {s: i for i, s in enumerate(P.times)}
    for r in pd.read_csv(REL / "task2" / p / "private" / "window_index.csv").itertuples():
        if r.condition != "queue_onset":
            continue
        t = ti[r.forecast_origin]; Pt = asof(P, t); q = (Pt.V <= Pt.vcut[None, :]) & np.isfinite(Pt.V)
        seen = np.zeros(P.n, bool); nd = 0
        for back in (1, 2, 3):
            dd = t - back * 288
            if dd - 6 >= 0:
                seen |= q[dd - 6:dd + 12].any(axis=0); nd += 1
        ev = seen.copy()
        for o in (1, 2):
            ev[o:] |= seen[:-o]; ev[:-o] |= seen[o:]
        S = {}
        for B in (0.25, 1.0):
            pb = probs[r.window_id].copy()
            if nd:
                pb[-1] = pb[-1] * np.where(ev, 1.0, B)
            S[B] = best_set(pb.ravel(), np.ones(pb.size, bool)).reshape(pb.shape)
        qs = Q.queue_pred.to_numpy()[QI[r.window_id]].reshape(S[0.25].shape[0], P.n).astype(bool); bad += int(not np.array_equal(S[0.25][-1], qs[-1]))
        d = int((S[1.0][-1] != S[0.25][-1]).sum()); nw += d > 0; nc += d
        out[r.window_id] = dict(s025=S[0.25][-1], s10=S[1.0][-1], nd=nd, changed=d, penalised=int((~ev).sum()) if nd else 0)
pickle.dump(out, open(W + "g290_april_b7s.pkl", "wb"))
print(f"April onset windows {len(out)} | identity B 0.25 against B7's queue T+30: {len(out) - bad} / {len(out)} | B 1.0 changes {nw} windows, {nc} cells "
      f"(added {sum(int((v['s10'] & ~v['s025']).sum()) for v in out.values())}, removed {sum(int((v['s025'] & ~v['s10']).sum()) for v in out.values())}) | windows with earlier days {sum(v['nd'] > 0 for v in out.values())}", flush=True)
