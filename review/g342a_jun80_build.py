"""G342a (11/05, ledger #267b inputs): the 80 official June 2030 windows (task2/<panel>/train/window_index.csv) as G320 builds windows
(history as released, MASKHIST=0, TFB_T2_NOT=1, TFB_T2_ALLFIN=1) -> NN/nn_test_jun80 with window_id = str(NWIN + i) (NWIN = the evaluator's
window count, featcache2_noT_af), plus work/t2/g342_jun80.pkl: the appended META rows and the labels (queue_matrix rows t+1..t+6, P.E rows
t+1..t+6 as t2_official_check) and the official window ids. Checks the labels against the evaluator's LAB on the windows that match.
usage: py -B -X utf8 g342a_jun80_build.py"""
import os, sys, subprocess, pickle, numpy as np, pandas as pd
sys.path.insert(0, r"C:/Users/user/tfb/src")
NN = "C:/tfb_data/tfb_nn/"; W = "C:/Users/user/tfb/work/t2/"
Z = np.load(W + "featcache2_noT_af.npz", allow_pickle=True); NWIN = len(Z["panel"]); LAB = pickle.load(open(W + "labels_all2_noT_af.pkl", "rb"))
from cachelib import load, T2_PANELS
from common import REL
from t2_windows import queue_matrix
rows = []
for p in T2_PANELS:
    P = load(p, "train"); Qm = queue_matrix(P); ti = {s: i for i, s in enumerate(P.times)}
    for r in pd.read_csv(REL / "task2" / p / "train" / "window_index.csv").itertuples():
        t = ti[r.forecast_origin]; rows.append(dict(wid=r.window_id, panel=p, t=t, cond=r.condition, date=P.times[t][:10], lab=Qm[t + 1:t + 7].copy(), el=P.E[t + 1:t + 7].copy()))
    load.cache_clear()
for i, r in enumerate(rows):
    r["k"] = NWIN + i
key = {(str(Z["panel"][k]), int(Z["t"][k]), str(Z["cond"][k])): k for k in range(NWIN) if str(Z["date"][k])[:7] == "2030-06"}
chk = [(r, key[(r["panel"], r["t"], r["cond"])]) for r in rows if (r["panel"], r["t"], r["cond"]) in key]
bad = sum(not (np.array_equal(r["lab"].astype(bool), LAB[k][0].astype(bool)) and np.array_equal(r["el"].astype(bool), LAB[k][1].astype(bool))) for r, k in chk)
print(f"NWIN {NWIN}; official June windows {len(rows)}; matched in the evaluator {len(chk)}, label mismatches {bad}; date format sample {Z['date'][0]!r}", flush=True)
pickle.dump(dict(NWIN=NWIN, rows=rows, match={r["k"]: k for r, k in chk}), open(W + "g342_jun80.pkl", "wb"))
os.environ["TFB_T2_NOT"] = "1"; os.environ["TFB_T2_ALLFIN"] = "1"; os.environ["TFB_T2_MASKHIST"] = "0"
code = f'''
import sys, os, numpy as np, warnings; warnings.filterwarnings("ignore"); sys.path.insert(0, r"C:/Users/user/tfb/src")
from cachelib import load
from t2_model import PanelInputs
from t2_cache import window_X
from t2_arrival import arrival_features, _dmin
from t2_demand import demand_features, profile
KS = {[r["k"] for r in rows]}; pan = {[r["panel"] for r in rows]}; t_ = {[r["t"] for r in rows]}; cnd = {[r["cond"] for r in rows]}; names = [str(x) for x in np.load("{NN}nn_data_afdem/names.npy", allow_pickle=True)]
XS, OF = [], [0]; PIc = {{}}
for k, p, t, cd in zip(KS, pan, t_, cnd):
    if p not in PIc:
        P = load(p, "train"); PIc[p] = (P, PanelInputs(P), profile(P))
    P, PI, prof = PIc[p]; Xw, _ = window_X(PI, t, str(cd)); Xw[:, names.index("prior_on")] = 0.0; Xw[:, names.index("prior_og")] = 0.0; V, Q, tl = PI.window(t); Xa, _ = arrival_features(P, V, Q, tl, PI.x); Xd = demand_features(P, Q, tl, t, P.times[t][:10], prof, _dmin(P, PI.x))
    X = np.column_stack([Xw, Xa, Xd]).astype(np.float32); XS.append(X); OF.append(OF[-1] + len(X))
o = "{NN}nn_test_jun80/"; os.makedirs(o, exist_ok=True); np.save(o + "X.npy", np.vstack(XS)); np.save(o + "off.npy", np.array(OF)); np.save(o + "window_id.npy", np.array([str(k) for k in KS])); np.save(o + "cond.npy", np.array(cnd))
print("built", len(KS), "windows, rows", OF[-1], "columns", XS[0].shape[1], "names", len(names), flush=True)
'''
subprocess.run([sys.executable, "-B", "-c", code], check=True, env=dict(os.environ))
