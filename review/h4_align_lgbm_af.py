"""Harness review H4 = audit a1 re-pointed at the ALLFIN (af) caches used by v14af / s23 (mp block is not used by v14 and is checked against the old noT cache only)."""
"""Audit A1: does the inference path (t2_predict.py, env as run_final_v13.cmd) rebuild exactly the training
matrix row (t2_fast.win_X from the caches) for the same train window?  Also checks the label alignment."""
import os, sys, pickle, numpy as np, warnings
warnings.filterwarnings("ignore")
os.environ.update(TFB_T2_NOT="1", TFB_T2_ALLFIN="1")
sys.path.insert(0, r"C:/Users/user/tfb/src"); sys.path.insert(0, r"C:/Users/user/tfb/audit")
from npzmm import npz_memmap
from cachelib import load
from t2_model import PanelInputs
from t2_cache import window_X
from t2_monthprior import features as mpf
from t2_arrival import arrival_features, _dmin
from t2_demand import demand_features, profile
from t2_windows import queue_matrix, F

W = r"C:/Users/user/tfb/work/t2/"
ZP = W + "featcache2_noT_af.npz"
Z = np.load(ZP, allow_pickle=True)
off, names = Z["off"], list(Z["names"]); meta = {k: Z[k] for k in ("panel", "t", "cond", "date", "is_eval", "is_sel")}
X_ALL = npz_memmap(ZP, "X")
MP = np.load(W + "mpcache2_noT.npy", mmap_mode="r"); ARR = np.load(W + "arrcache2_noT_af.npy", mmap_mode="r"); DEM = np.load(W + "demcache2_noT_af.npy", mmap_mode="r")
LAB = pickle.load(open(W + "labels_all2_noT_af.pkl", "rb"))
print("featcache rows", X_ALL.shape, "windows", len(off) - 1, "is_sel", int(meta["is_sel"].sum()), "is_eval", int(meta["is_eval"].sum()))
rng = np.random.default_rng(0)
panels = sys.argv[1:] or ["D7_I405_N", "D12_I5_S"]
for p in panels:
    P = load(p, "train"); PI = PanelInputs(P); Qm = queue_matrix(P)
    ks = np.flatnonzero(meta["panel"] == p)
    pick = np.concatenate([rng.choice(ks[meta["is_sel"][ks]], 8, replace=False), rng.choice(ks[~meta["is_sel"][ks]], 8, replace=False)])
    worst = {}
    for k in pick:
        t = int(meta["t"][k]); a, b = off[k], off[k + 1]
        Xb, nb = window_X(PI, t, str(meta["cond"][k]))
        assert nb == names, "column names differ"
        base = np.asarray(X_ALL[a:b]).copy()
        ip = [names.index("prior_on"), names.index("prior_og")]; Xb[:, ip] = base[:, ip]   # priors filled later in both paths
        Xm = mpf(p, "train", t)
        Vw, Qw, tlw = PI.window(t); Xa, _ = arrival_features(P, Vw, Qw, tlw, PI.x)
        Xd_all = demand_features(P, Qw, tlw, t, P.times[t][:10], profile(load(p, "train")), _dmin(P, PI.x))
        Xd_lomo = demand_features(P, Qw, tlw, t, P.times[t][:10], profile(P, str(meta["date"][k])[:7]), _dmin(P, PI.x))
        for nm, A, B in (("base", Xb, base), ("mp", Xm, MP[a:b]), ("arr", Xa, ARR[a:b]), ("dem_lomo", Xd_lomo, DEM[a:b]), ("dem_allprof_vs_cache", Xd_all, DEM[a:b])):
            A = np.asarray(A, np.float64); B = np.asarray(B, np.float64)
            same_nan = np.array_equal(np.isnan(A), np.isnan(B))
            d = np.nanmax(np.abs(A - B)) if np.isfinite(A).any() else 0.0
            worst.setdefault(nm, []).append((d, same_nan))
        lab, el = LAB[k]
        assert np.array_equal(lab, Qm[t + 1:t + F + 1]) and np.array_equal(el, P.E[t + 1:t + F + 1]), f"label misalignment k={k}"
    for nm, v in worst.items():
        print(p, f"{nm:22s} max|diff| {max(x[0] for x in v):.3g}  NaN-pattern equal in {sum(x[1] for x in v)}/{len(v)} windows")
    print(p, "labels aligned for", len(pick), "windows")
    del P, PI, Qm
    load.cache_clear()
