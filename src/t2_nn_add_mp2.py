"""Append the causal within-month evidence v2 (t2_monthprior2: earlier days of the same month only, release blanks
reproduced on train, queue heads instead of the first queue of the day) to the af+dem NN inputs:
nn_data_afdem -> nn_data_afdemmp2 (train, rows in featcache2_noT_af order) and nn_test_afdem -> nn_test_afdemmp2
(validation/private, nn_test window_id order). 80 + 6 columns.

Why (CAUSAL_MAP P1, 10/10): the v1 month prior did nothing because the model also had location memory (position,
priors, static fingerprint) that wins inside the train months. A model without memory (ZERO=pos+tod+dow+statics) can
only locate this month's bottlenecks through this evidence, which stays valid in the test months.
"""
import os, shutil, time, numpy as np, pandas as pd
os.environ.setdefault("TFB_T2_NOT", "1")
from cachelib import load
from common import REL
from t2_monthprior2 import features as mpf, NAMES

W = "C:/Users/user/tfb/work/t2/"; t0 = time.time()
Z = np.load(W + "featcache2_noT_af.npz", allow_pickle=True); off, panel, tt = Z["off"], Z["panel"], Z["t"]
SUF = os.environ.get("TFB_MP2_SUF", "mp2")                  # mp3 = evidence capped to 3 earlier days (TFB_MP2_MAXDAYS=3)
for src, dst in (("E:/tfb_nn/nn_data_afdem", f"E:/tfb_nn/nn_data_afdem{SUF}"), ("E:/tfb_nn/nn_test_afdem", f"E:/tfb_nn/nn_test_afdem{SUF}")):
    os.makedirs(dst, exist_ok=True)
    X = np.load(f"{src}/X.npy", mmap_mode="r"); offs = np.load(f"{src}/off.npy")
    MP = np.zeros((len(X), len(NAMES)), np.float32)
    if "nn_data" in src:
        assert (offs == off).all(), "nn_data rows differ from featcache2_noT_af"
        for k in range(len(off) - 1):
            MP[off[k]:off[k + 1]] = mpf(str(panel[k]), "train", int(tt[k]))
            if k % 2000 == 0:
                print("train", k, f"{time.time()-t0:.0f}s", flush=True)
    else:
        wid = np.load(f"{src}/window_id.npy", allow_pickle=True).astype(str); TI = {}
        for i, w in enumerate(wid):
            split = "validation" if "_validation_" in w else "private"; p = w[3:w.index("_" + split + "_")]
            if (p, split) not in TI:
                P = load(p, split); ti = {s: j for j, s in enumerate(P.times)}
                wi = pd.read_csv(REL / "task2" / p / split / "window_index.csv")
                TI[(p, split)] = (P.n, {r.window_id: ti[r.forecast_origin] for r in wi.itertuples()})
            n, tmap = TI[(p, split)]
            assert offs[i + 1] - offs[i] == n, ("links", w)
            MP[offs[i]:offs[i + 1]] = mpf(p, split, tmap[w])
    out = np.lib.format.open_memmap(f"{dst}/X.npy", mode="w+", dtype=np.float32, shape=(len(X), X.shape[1] + MP.shape[1]))
    for a in range(0, len(X), 1 << 21):
        out[a:a + (1 << 21), :X.shape[1]] = X[a:a + (1 << 21)]
        out[a:a + (1 << 21), X.shape[1]:] = MP[a:a + (1 << 21)]
    out.flush(); del out
    for f in os.listdir(src):
        if f not in ("X.npy", "names.npy"):
            shutil.copyfile(f"{src}/{f}", f"{dst}/{f}")
    names = list(np.load("E:/tfb_nn/nn_data_afdem/names.npy", allow_pickle=True)) + NAMES
    np.save(f"{dst}/names.npy", np.array(names))
    print("ok", dst, (len(X), X.shape[1] + MP.shape[1]), f"finite share of mp2 {np.isfinite(MP[:, 0]).mean():.3f}", f"{time.time()-t0:.0f}s", flush=True)
