"""Rebuild the Task 2 caches with every finite released value as input (audit F4), same windows/rows/labels:
featcache2_noT_af.npz, labels_all2_noT_af.pkl, arrcache2_noT_af.npy, demcache2_noT_af.npy."""
import os, shutil, time, warnings, numpy as np
os.environ["TFB_T2_NOT"] = "1"; os.environ["TFB_T2_ALLFIN"] = "1"; warnings.filterwarnings("ignore")
from cachelib import load
from t2_model import PanelInputs
from t2_cache import window_X
from t2_arrival import arrival_features, _dmin
from t2_demand import demand_features, profile

W = "C:/Users/user/tfb/work/t2/"
Z = np.load(W + "featcache2_noT.npz", allow_pickle=True)
off, panel, tt, cond, date = Z["off"], Z["panel"], Z["t"], Z["cond"], Z["date"]
X = np.zeros((off[-1], Z["X"].shape[1]), np.float32); A = np.zeros((off[-1], 7), np.float32); Dm = np.zeros((off[-1], 6), np.float32)
PIs = {}; t0 = time.time()
for k in range(len(off) - 1):
    p = str(panel[k])
    if p not in PIs:
        PIs[p] = PanelInputs(load(p, "train"))
    PI = PIs[p]; P = PI.P; t = int(tt[k]); a, b = off[k], off[k + 1]
    X[a:b], _ = window_X(PI, t, str(cond[k]))
    V, Q, tl = PI.window(t)
    A[a:b], _ = arrival_features(P, V, Q, tl, PI.x)
    Dm[a:b] = demand_features(P, Q, tl, t, str(date[k])[:10], profile(P, str(date[k])[:7]), _dmin(P, PI.x))
    if k % 10000 == 0:
        print(k, f"{time.time()-t0:.0f}s", flush=True)
np.savez(W + "featcache2_noT_af.npz", X=X, **{k: Z[k] for k in Z.files if k != "X"})
shutil.copyfile(W + "labels_all2_noT.pkl", W + "labels_all2_noT_af.pkl")
np.save(W + "arrcache2_noT_af.npy", A); np.save(W + "demcache2_noT_af.npy", Dm)
print("done", f"{time.time()-t0:.0f}s", "| NaN share new %.4f" % np.isnan(X[:2_000_000]).mean())
