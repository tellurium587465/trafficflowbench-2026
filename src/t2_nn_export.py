"""Export Task 2 per-window link-feature matrices (+ labels, metadata) for the WSL/GPU neural net.

Writes work/t2/nn_data.npz: X (rows, F) float32 aligned with featcache2_noT rows (base + arrival features,
priors filled with all-train values), off, panel/t/cond/date/is_eval/is_sel, labels Y (rows, 6) uint8 and
eligibility EL (rows, 6) uint8 per link row.
"""
import os, numpy as np
os.environ.setdefault("TFB_T2CACHE", "2_noT")
os.environ.setdefault("TFB_T2_ARR", "1")
import t2_fast as T

pr = T.priors(set())
rows = []
X = np.empty((T.OFF[-1], len(T.NAMES) + T.ARR_ALL.shape[1]), np.float32)
Y = np.zeros((T.OFF[-1], 6), np.uint8); EL = np.zeros((T.OFF[-1], 6), np.uint8)
for k in range(T.NWIN):
    a, b = T.OFF[k], T.OFF[k + 1]
    X[a:b] = T.win_X(k, pr)
    lab, el = T.LAB[k]
    Y[a:b] = lab.T; EL[a:b] = el.T
np.savez(r"C:/Users/user/tfb/work/t2/nn_data.npz", X=X, Y=Y, EL=EL, off=T.OFF,
         panel=T.META["panel"], t=T.META["t"], cond=T.META["cond"], date=T.META["date"],
         is_eval=T.META["is_eval"], is_sel=T.META["is_sel"], names=np.array(T.NAMES + [f"arr{i}" for i in range(T.ARR_ALL.shape[1])]))
print("exported", X.shape)
