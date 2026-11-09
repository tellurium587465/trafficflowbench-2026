"""Append the causal within-month location priors (t2_monthprior: earlier days of the same month only) to the
NN inputs: nn_data_dem -> nn_data_demmp (train, from mpcache) and nn_test_dem -> nn_test_demmp (val/private)."""
import os, shutil, numpy as np, pandas as pd
os.environ.setdefault("TFB_T2_NOT", "1")
from cachelib import load, T2_PANELS
from common import REL
from t2_monthprior import features as mpf, NAMES

for src, dst in (("E:/tfb_nn/nn_data_dem", "E:/tfb_nn/nn_data_demmp"), ("E:/tfb_nn/nn_test_dem", "E:/tfb_nn/nn_test_demmp")):
    os.makedirs(dst, exist_ok=True)
    X = np.load(f"{src}/X.npy", mmap_mode="r")
    if "nn_data" in src:
        MP = np.load("C:/Users/user/tfb/work/t2/mpcache2_noT.npy", mmap_mode="r")
    else:
        rows = []
        for split in ("validation", "private"):
            for p in T2_PANELS:
                P = load(p, split); ti = {s: i for i, s in enumerate(P.times)}
                for r in pd.read_csv(REL / "task2" / p / split / "window_index.csv").itertuples():
                    rows.append(mpf(p, split, ti[r.forecast_origin]))
        MP = np.vstack(rows)
    assert len(MP) == len(X), (MP.shape, X.shape)
    out = np.lib.format.open_memmap(f"{dst}/X.npy", mode="w+", dtype=np.float32, shape=(len(X), X.shape[1] + MP.shape[1]))
    for a in range(0, len(X), 1 << 21):
        out[a:a + (1 << 21), :X.shape[1]] = X[a:a + (1 << 21)]
        out[a:a + (1 << 21), X.shape[1]:] = MP[a:a + (1 << 21)]
    out.flush()
    for f in os.listdir(src):
        if f != "X.npy":
            shutil.copyfile(f"{src}/{f}", f"{dst}/{f}")
    if os.path.exists(f"{src}/names.npy"):
        np.save(f"{dst}/names.npy", np.array(list(np.load(f"{src}/names.npy", allow_pickle=True)) + NAMES))
    print("ok", dst, out.shape)
