"""G71 (strategy §16.5, 10/22): NN inputs as within-window percentiles (domain generalisation: no absolute level, no link
fingerprint can be memorised from the values). Copies of the data / test dirs where every non-memory column (the G69
columns: all but the ZERO list, priors, arr5, cond) is replaced by its within-window percentile (rank - 0.5) / n over the
finite values; NaN stays NaN (the NN's missing mask is unchanged); memory columns are left as they are (ZERO zeroes them).
  train : nn_data_afdem -> nn_data_afdem_pct ;  m3 : nn_test_m3 -> nn_test_m3_pct ;  test : nn_test_afdem -> nn_test_afdem_pct
usage: python g71_pct_data.py <train|m3|test>"""
import os, sys, shutil, time
sys.path.insert(0, r"C:/Users/user/tfb/src")
import numpy as np
from scipy.stats import rankdata
from common import NN

kind = sys.argv[1]; t0 = time.time()
SRC = {"train": "nn_data_afdem", "m3": "nn_test_m3", "test": "nn_test_afdem"}[kind]; DST = SRC + "_pct"
NAMES = [str(x) for x in np.load(f"{NN}/nn_data_afdem/names.npy", allow_pickle=True)]
DROP = {"prior_on", "prior_og", "pos", "vf", "capl", "lanes", "length", "w", "tod", "dow", "arr5", "cond"}
COLS = [i for i, n in enumerate(NAMES) if n not in DROP]
os.makedirs(f"{NN}/{DST}", exist_ok=True)
for f in os.listdir(f"{NN}/{SRC}"):
    shutil.copyfile(f"{NN}/{SRC}/{f}", f"{NN}/{DST}/{f}")
X = np.load(f"{NN}/{DST}/X.npy", mmap_mode="r+"); off = np.load(f"{NN}/{DST}/off.npy")
for i in range(len(off) - 1):
    a, b = off[i], off[i + 1]; Xw = np.asarray(X[a:b][:, COLS], dtype=np.float64); P = np.full_like(Xw, np.nan)
    for c in range(Xw.shape[1]):
        v = Xw[:, c]; ok = np.isfinite(v)
        if ok.any():
            P[ok, c] = (rankdata(v[ok]) - 0.5) / ok.sum()
    X[a:b, COLS] = P.astype(np.float32)
    if i % 10000 == 0:
        print(f"{i}/{len(off) - 1} {time.time() - t0:.0f}s", flush=True)
X.flush(); print(f"{kind}: {len(off) - 1} windows, {len(COLS)} columns -> within-window percentiles -> {DST} ({time.time() - t0:.0f}s)")
