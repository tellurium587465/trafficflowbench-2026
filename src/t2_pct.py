"""Within-window percentile inputs (11/03 ledger #201, #17 / #172b; the transform of review/g71_pct_data.py, keep in sync): every
non-memory column (all but the DROP list) of a test directory's X is replaced by its within-window percentile (rank - 0.5) / n over the
finite values; NaN stays NaN; the other files are copied. The rank-input network pcL (review/t2_nn3.py on nn_data_afdem_pct, fold 2)
reads these. usage: python t2_pct.py <src dir under NN> <dst dir under NN>"""
import os, sys, shutil, numpy as np
from scipy.stats import rankdata
from common import NN

DROP = {"prior_on", "prior_og", "pos", "vf", "capl", "lanes", "length", "w", "tod", "dow", "arr5", "cond"}


def pct_dir(src, dst):
    names = [str(x) for x in np.load(f"{NN}/nn_data_afdem/names.npy", allow_pickle=True)]; cols = [i for i, n in enumerate(names) if n not in DROP]
    os.makedirs(f"{NN}/{dst}", exist_ok=True)
    for f in os.listdir(f"{NN}/{src}"):
        shutil.copyfile(f"{NN}/{src}/{f}", f"{NN}/{dst}/{f}")
    X = np.load(f"{NN}/{dst}/X.npy", mmap_mode="r+"); off = np.load(f"{NN}/{dst}/off.npy")
    for i in range(len(off) - 1):
        a, b = off[i], off[i + 1]; Xw = np.asarray(X[a:b][:, cols], dtype=np.float64); P = np.full_like(Xw, np.nan)
        for c in range(Xw.shape[1]):
            v = Xw[:, c]; ok = np.isfinite(v)
            if ok.any():
                P[ok, c] = (rankdata(v[ok]) - 0.5) / ok.sum()
        X[a:b, cols] = P.astype(np.float32)
    X.flush(); del X


if __name__ == "__main__":
    pct_dir(sys.argv[1], sys.argv[2])
