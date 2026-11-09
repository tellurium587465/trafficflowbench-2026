"""Demand-profile features for every cached train window (aligned with featcache rows), leave-one-month-out."""
import os, time, warnings, numpy as np
from cachelib import load
from t2_model import PanelInputs
from t2_arrival import _dmin
from t2_demand import profile, demand_features, NAMES
warnings.filterwarnings("ignore")
SUF = os.environ.get("TFB_T2CACHE", "2_noT")
Z = np.load(rf"C:/Users/user/tfb/work/t2/featcache{SUF}.npz", allow_pickle=True)
off = Z["off"]; panel = Z["panel"]; t = Z["t"]; date = Z["date"]
PIs = {}; out = np.zeros((off[-1], len(NAMES)), np.float32); t0 = time.time()
for k in range(len(off) - 1):
    p = str(panel[k])
    if p not in PIs:
        PIs[p] = PanelInputs(load(p, "train"))
    PI = PIs[p]; P = PI.P; tk = int(t[k])
    V, Q, tl = PI.window(tk)
    out[off[k]:off[k + 1]] = demand_features(P, Q, tl, tk, str(date[k])[:10], profile(P, str(date[k])[:7]), _dmin(P, PI.x))
    if k % 10000 == 0:
        print(k, f"{time.time()-t0:.0f}s", flush=True)
np.save(rf"C:/Users/user/tfb/work/t2/demcache{SUF}.npy", out)
print("done", out.shape, NAMES, f"{time.time()-t0:.0f}s")
