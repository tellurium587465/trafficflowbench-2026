"""G272 (11/04, ledger #228 stage 2', declared by the strategy before the results): January / February ongoing windows chosen the way
the official windows are reproduced (auditor's rep5.py scan2, 36/40): per corridor and month, the origins in time order; an origin is
taken when it is >= 72 rows after the last taken origin (onset or ongoing), its history and horizon lie on one calendar day, the history
T-12 .. T-1 is >= 70 % observed, the horizon T+1 .. T+6 holds a queue (raw), and either the history has no queue (an onset window, kept
only for the spacing) or it holds >= 2 queued cells with >= 2 on one link and the persistence IoU of the 'and'-filled state at T against
T+1 .. T+6 over ever-finite links is <= 0.9 (an ongoing window). No cap on the count (the instrument wants every such window).
  select: writes work/t2/g272_windows.csv
  build : the networks' inputs for these windows as G220 builds the January / February ones (unmasked history, no row at T, priors 0),
          C:/tfb_data/tfb_nn/nn_test_jfnew/ (window ids "panel|t")
usage: py -B -X utf8 g272_jf_firstpass.py select | build"""
import os, sys, subprocess, numpy as np, pandas as pd
sys.path.insert(0, r"C:/Users/user/tfb/src"); MODE = sys.argv[1]; W = "C:/Users/user/tfb/work/t2/"; NN = "C:/tfb_data/tfb_nn/"; H, F = 12, 6
if MODE == "select":
    from cachelib import load, T2_PANELS

    def impute_and(V, vcut, E, maxgap=6):
        q = np.where(np.isfinite(V), (V <= vcut[None, :]).astype(np.float32), np.nan); df = pd.DataFrame(q)
        f = df.ffill(limit=maxgap).to_numpy(); b = df.bfill(limit=maxgap).to_numpy()
        est = np.where(np.isfinite(f) & np.isfinite(b), np.minimum(f, b), np.where(np.isfinite(f), f, b)); out = np.where(np.isfinite(q), q, est)
        return (np.nan_to_num(out, nan=0.0) >= 0.5) & E

    def piou(base, fut, el):
        p = np.broadcast_to(base, fut.shape) & el; t = fut & el; u = (p | t).sum(); return 1.0 if u == 0 else (p & t).sum() / u

    rows = []
    for p in T2_PANELS:
        P = load(p, "train"); obs = np.isfinite(P.V) & P.E; ever = np.isfinite(P.V).any(axis=0); Lraw = (np.nan_to_num(P.V, nan=1e9) <= P.vcut[None, :]) & obs
        Lp = impute_and(P.V, P.vcut, P.E) & ever[None, :]; anyh = Lraw.any(axis=1); mon = np.array([s[:7] for s in P.times])
        for mo in ("2031-01", "2031-02"):
            rr = np.flatnonzero(mon == mo); last = -10 ** 9
            for t in range(max(int(rr[0]), H), int(rr[-1]) - F):
                if t - last < 72 or P.times[t - H][:10] != P.times[t + F][:10] or obs[t - H:t].mean() < 0.7 or not anyh[t + 1:t + F + 1].any():
                    continue
                hq = Lraw[t - H:t]
                if not hq.any():
                    if not anyh[t]:
                        last = t
                    continue
                if not (hq.sum() >= 2 and hq.sum(axis=0).max() >= 2):
                    continue
                pi = piou(Lp[t], Lp[t + 1:t + F + 1], np.broadcast_to(ever, (F, len(ever))))
                if pi <= 0.9:
                    rows.append(dict(panel=p, t=t, month=mo, day=int(P.times[t][8:10]) - 1, hour=int(P.times[t][11:13]), piou=pi)); last = t
        load.cache_clear(); print(p, sum(r["panel"] == p for r in rows), flush=True)
    D = pd.DataFrame(rows); D.to_csv(W + "g272_windows.csv", index=False); print(len(D), "ongoing windows | by month", D.month.value_counts().to_dict(), "| hour median", D.hour.median())
    sys.exit()
assert MODE == "build"
D = pd.read_csv(W + "g272_windows.csv"); d = NN + "nn_data_afdem/"
os.environ["TFB_T2_NOT"] = "1"; os.environ["TFB_T2_ALLFIN"] = "1"; os.environ["TFB_T2_MASKHIST"] = "0"
code = f'''
import sys, numpy as np, warnings; warnings.filterwarnings("ignore"); sys.path.insert(0, r"C:/Users/user/tfb/src")
from cachelib import load
from t2_model import PanelInputs
from t2_cache import window_X
from t2_arrival import arrival_features, _dmin
from t2_demand import demand_features, profile
pan = {D.panel.tolist()}; t_ = {[int(x) for x in D.t]}; names = [str(x) for x in np.load("{d}names.npy", allow_pickle=True)]
XS, OF = [], [0]; PIc = {{}}
for p, t in zip(pan, t_):
    if p not in PIc:
        P = load(p, "train"); PIc[p] = (P, PanelInputs(P), profile(P))
    P, PI, prof = PIc[p]; Xw, _ = window_X(PI, t, "queue_ongoing"); Xw[:, names.index("prior_on")] = 0.0; Xw[:, names.index("prior_og")] = 0.0; V, Q, tl = PI.window(t); Xa, _ = arrival_features(P, V, Q, tl, PI.x); Xd = demand_features(P, Q, tl, t, P.times[t][:10], prof, _dmin(P, PI.x)); X = np.column_stack([Xw, Xa, Xd]).astype(np.float32)
    XS.append(X); OF.append(OF[-1] + len(X))
import os
o = "{NN}nn_test_jfnew/"; os.makedirs(o, exist_ok=True); np.save(o + "X.npy", np.vstack(XS)); np.save(o + "off.npy", np.array(OF)); np.save(o + "window_id.npy", np.array([f"{{p}}|{{t}}" for p, t in zip(pan, t_)])); np.save(o + "cond.npy", np.array(["queue_ongoing"] * len(pan)))
print("built", len(pan), "windows", np.vstack(XS).shape, flush=True)
'''
subprocess.run([sys.executable, "-B", "-c", code], check=True, env=dict(os.environ))
