"""G320 (11/05, ledger #254 inputs): the June 2030 selector windows (both conditions) rebuilt as G296 (history as released with MASKHIST=0, or masked with 1) -> NN/nn_test_junmh<0|1>. usage: py -B -X utf8 g320_june_build.py [0|1]"""
import os, sys, subprocess, numpy as np
sys.dont_write_bytecode = True
NN = "C:/tfb_data/tfb_nn/"; d = NN + "nn_data_afdem/"; date = np.load(d + "date.npy", allow_pickle=True); cond = np.load(d + "cond.npy", allow_pickle=True); isel = np.load(d + "is_sel.npy")
pan = np.array([str(x) for x in np.load(d + "panel.npy", allow_pickle=True)]); t_ = np.load(d + "t.npy")
mon = np.array([str(x)[:7] for x in date]); KS = np.flatnonzero(isel & np.isin(mon, ("2030-06",)) & np.isin(cond, ("queue_onset", "queue_ongoing"))); print("June selector windows", len(KS), flush=True)
os.environ["TFB_T2_NOT"] = "1"; os.environ["TFB_T2_ALLFIN"] = "1"; MH = (sys.argv[1:] or ["0"])[0]; assert MH in ("0", "1"); os.environ["TFB_T2_MASKHIST"] = MH
code = f'''
import sys, os, numpy as np, warnings; warnings.filterwarnings("ignore"); sys.path.insert(0, r"C:/Users/user/tfb/src")
from cachelib import load
from t2_model import PanelInputs
from t2_cache import window_X
from t2_arrival import arrival_features, _dmin
from t2_demand import demand_features, profile
KS = np.array({KS.tolist()}); pan = {pan[KS].tolist()}; t_ = {t_[KS].tolist()}; cnd = {[str(x) for x in cond[KS]]}; names = [str(x) for x in np.load("{d}names.npy", allow_pickle=True)]
XS, OF = [], [0]; PIc = {{}}
for k, p, t, cd in zip(KS, pan, t_, cnd):
    if p not in PIc:
        P = load(p, "train"); PIc[p] = (P, PanelInputs(P), profile(P))
    P, PI, prof = PIc[p]; Xw, _ = window_X(PI, t, str(cd)); Xw[:, names.index("prior_on")] = 0.0; Xw[:, names.index("prior_og")] = 0.0; V, Q, tl = PI.window(t); Xa, _ = arrival_features(P, V, Q, tl, PI.x); Xd = demand_features(P, Q, tl, t, P.times[t][:10], prof, _dmin(P, PI.x))
    X = np.column_stack([Xw, Xa, Xd]).astype(np.float32); XS.append(X); OF.append(OF[-1] + len(X))
o = "{NN}nn_test_junmh" + os.environ["TFB_T2_MASKHIST"] + "/"; os.makedirs(o, exist_ok=True); np.save(o + "X.npy", np.vstack(XS)); np.save(o + "off.npy", np.array(OF)); np.save(o + "window_id.npy", np.array([str(k) for k in KS])); np.save(o + "cond.npy", np.array(cnd))
print("built", len(KS), "windows, rows", OF[-1], "columns", XS[0].shape[1], "names", len(names), flush=True)
'''
subprocess.run([sys.executable, "-B", "-c", code], check=True, env=dict(os.environ))
