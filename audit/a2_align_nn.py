"""Audit A2: NN inputs. (1) nn_data* rows == the cached LGBM features of the same window (column blocks);
(2) nn_test* rows == features rebuilt by the inference path for validation/private windows, same order as window_index."""
import os, sys, numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
os.environ.update(TFB_T2_NOT="1")
sys.path.insert(0, r"C:/Users/user/tfb/src"); sys.path.insert(0, r"C:/Users/user/tfb/audit")
from npzmm import npz_memmap
from cachelib import load, T2_PANELS
from common import REL
from t2_model import PanelInputs
from t2_cache import window_X
from t2_monthprior import features as mpf
from t2_arrival import arrival_features, _dmin
from t2_demand import demand_features, profile


def raw(PI, t):      # copy of t2_rawcache.raw (that module computes everything at import)
    V, Q, tl = PI.window(t); P = PI.P
    rows = np.arange(tl - 12, tl)
    v = np.full((12, P.n), np.nan, np.float32); q = np.full((12, P.n), np.nan, np.float32)
    ok = rows >= 0
    v[ok] = V[rows[ok]] / P.vf[None, :]; q[ok] = Q[rows[ok]] / P.cap[None, :]
    return np.column_stack([v.T, q.T]).astype(np.float32)


def cmp(A, B):
    A = np.asarray(A, np.float64); B = np.asarray(B, np.float64)
    nan_eq = np.array_equal(np.isnan(A), np.isnan(B))
    d = float(np.nanmax(np.abs(A - B))) if np.isfinite(A - B).any() else 0.0
    return d, nan_eq


W = r"C:/Users/user/tfb/work/t2/"
Z = np.load(W + "featcache2_noT.npz", allow_pickle=True); off = Z["off"]; names = list(Z["names"])
meta = {k: Z[k] for k in ("panel", "t", "cond", "date", "is_sel")}
XA = npz_memmap(W + "featcache2_noT.npz", "X")
ARR = np.load(W + "arrcache2_noT.npy", mmap_mode="r"); DEM = np.load(W + "demcache2_noT.npy", mmap_mode="r"); MP = np.load(W + "mpcache2_noT.npy", mmap_mode="r")
N = {d: np.load(f"E:/tfb_nn/{d}/X.npy", mmap_mode="r") for d in ("nn_data", "nn_data_dem", "nn_data_demmp", "nn_data_demraw")}
noff = np.load("E:/tfb_nn/nn_data/off.npy")
print("nn_data off == featcache off:", np.array_equal(noff, off))
ip = [names.index("prior_on"), names.index("prior_og")]
rng = np.random.default_rng(1)
ks = rng.choice(len(off) - 1, 40, replace=False)
res = {}
for k in ks:
    a, b = off[k], off[k + 1]
    base = np.asarray(XA[a:b]).copy(); nd = np.asarray(N["nn_data"][a:b]).copy()
    base[:, ip] = 0; nd[:, ip] = 0
    for nm, A, B in (("nn_data[:67]==featcache", nd[:, :67], base), ("nn_data[67:74]==arrcache", N["nn_data"][a:b, 67:74], ARR[a:b]),
                     ("dem[:74]==nn_data", N["nn_data_dem"][a:b, :74], N["nn_data"][a:b]), ("dem[74:80]==demcache", N["nn_data_dem"][a:b, 74:80], DEM[a:b]),
                     ("demmp[:80]==dem", N["nn_data_demmp"][a:b, :80], N["nn_data_dem"][a:b]), ("demmp[80:84]==mpcache", N["nn_data_demmp"][a:b, 80:84], MP[a:b]),
                     ("demraw[:80]==dem", N["nn_data_demraw"][a:b, :80], N["nn_data_dem"][a:b])):
        res.setdefault(nm, []).append(cmp(A, B))
for nm, v in res.items():
    print(f"train {nm:28s} max|diff| {max(x[0] for x in v):.3g} NaN-eq {sum(x[1] for x in v)}/{len(v)}")
# raw block vs recomputation for a few windows of two panels
for p in ("D7_I210_E",):
    P = load(p, "train"); PI = PanelInputs(P)
    kk = rng.choice(np.flatnonzero(meta["panel"] == p), 10, replace=False)
    r = [cmp(N["nn_data_demraw"][off[k]:off[k + 1], 80:104], raw(PI, int(meta["t"][k]))) for k in kk]
    print("train raw block", p, "max|diff|", max(x[0] for x in r), "NaN-eq", sum(x[1] for x in r), "/", len(r))
    del P, PI; load.cache_clear()
# test side
T = {d: np.load(f"E:/tfb_nn/{d}/X.npy", mmap_mode="r") for d in ("nn_test", "nn_test_dem", "nn_test_demmp", "nn_test_demraw")}
toff = np.load("E:/tfb_nn/nn_test/off.npy"); wid = np.load("E:/tfb_nn/nn_test/window_id.npy", allow_pickle=True)
for d in ("nn_test_dem", "nn_test_demmp", "nn_test_demraw"):
    print(d, "window_id identical to nn_test:", np.array_equal(np.load(f"E:/tfb_nn/{d}/window_id.npy", allow_pickle=True), wid),
          "off identical:", np.array_equal(np.load(f"E:/tfb_nn/{d}/off.npy"), toff))
print("prefix checks test:", cmp(T["nn_test_dem"][:, :74], T["nn_test"][:]), cmp(T["nn_test_demmp"][:, :80], T["nn_test_dem"][:]), cmp(T["nn_test_demraw"][:, :80], T["nn_test_dem"][:]))
order = []
for split in ("validation", "private"):
    for p in T2_PANELS:
        for r in pd.read_csv(REL / "task2" / p / split / "window_index.csv").itertuples():
            order.append((split, p, r.window_id, r.forecast_origin, r.condition))
print("test window order == window_index order:", [str(w) for w in wid] == [o[2] for o in order])
res = {}
for split in ("validation", "private"):
    for p in ("D7_I405_S", "D12_I5_N"):
        P = load(p, split); PI = PanelInputs(P); ti = {s: i for i, s in enumerate(P.times)}; prof = profile(load(p, "train"))
        for i, o in enumerate(order):
            if o[0] != split or o[1] != p:
                continue
            t = ti[o[3]]; a, b = toff[i], toff[i + 1]
            Xb, _ = window_X(PI, t, o[4]); Vw, Qw, tlw = PI.window(t); Xa, _ = arrival_features(P, Vw, Qw, tlw, PI.x)
            Xd = demand_features(P, Qw, tlw, t, P.times[t][:10], prof, _dmin(P, PI.x)); Xm = mpf(p, split, t); Xr = raw(PI, t)
            got = np.asarray(T["nn_test_demmp"][a:b]).copy(); got[:, ip] = 0; Xb[:, ip] = 0
            for nm, A, B in (("base", got[:, :67], Xb), ("arr", got[:, 67:74], Xa), ("dem", got[:, 74:80], Xd), ("mp", got[:, 80:84], Xm),
                             ("raw", T["nn_test_demraw"][a:b, 80:104], Xr)):
                res.setdefault(nm, []).append(cmp(A, B))
        del P, PI; load.cache_clear()
for nm, v in res.items():
    print(f"test {nm:5s} max|diff| {max(x[0] for x in v):.3g} NaN-eq {sum(x[1] for x in v)}/{len(v)}")
# NaN rate per column, train vs test (a dropped/never-filled column would show up here)
nm_all = list(np.load("E:/tfb_nn/nn_data_demraw/names.npy", allow_pickle=True))
sub = np.asarray(N["nn_data_demraw"][::97]); tst = np.asarray(T["nn_test_demraw"][:])
fr_tr = np.isnan(sub).mean(0); fr_te = np.isnan(tst).mean(0)
bad = [(nm_all[j], round(fr_tr[j], 3), round(fr_te[j], 3)) for j in range(len(nm_all)) if abs(fr_tr[j] - fr_te[j]) > 0.05]
print("columns whose NaN rate differs >5pt train vs test:", bad)
