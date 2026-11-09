"""Sim-to-real training data: the real NN inputs (E:/tfb_nn/nn_data_afdem, 57,894 real train windows) + the
randomized twin worlds' training windows (nn_data_twinAR rows whose date is in a synthetic year >= 2040; world A
windows are NOT added). Synthetic windows carry is_eval = is_sel = False and synthetic months, so in the month-CV
they are always on the training side and every score is on real windows only (the in-distribution check stays
clean). -> E:/tfb_nn/nn_data_afdemsimR (same 80 columns; asserted)."""
import os, numpy as np
R = "E:/tfb_nn/nn_data_afdem"; S = "E:/tfb_nn/nn_data_twinAR24"; D = "E:/tfb_nn/nn_data_afdemsimR24"
os.makedirs(D, exist_ok=True)
nr = [str(x) for x in np.load(f"{R}/names.npy", allow_pickle=True)]; ns = [str(x) for x in np.load(f"{S}/names.npy", allow_pickle=True)]
assert nr == ns, "column names differ"
XR = np.load(f"{R}/X.npy", mmap_mode="r"); XS = np.load(f"{S}/X.npy", mmap_mode="r")
oR = np.load(f"{R}/off.npy"); oS = np.load(f"{S}/off.npy")
dS = np.load(f"{S}/date.npy"); keep = np.array([d[:4] >= "2040" for d in dS])
ks = np.flatnonzero(keep); print("synthetic windows added", len(ks), "of", len(dS))
rows = np.concatenate([np.arange(oS[k], oS[k + 1]) for k in ks])
nR, nS = len(XR), len(rows)
X = np.lib.format.open_memmap(f"{D}/X.npy", mode="w+", dtype=np.float32, shape=(int(nR + nS), int(XR.shape[1])))
C = 1 << 21
for a in range(0, nR, C):
    b = min(a + C, nR); X[a:b] = XR[a:b]
for a in range(0, nS, C):
    b = min(a + C, nS); X[nR + a:nR + b] = XS[rows[a:b]]
X.flush(); del X
lens = oS[ks + 1] - oS[ks]
np.save(f"{D}/off.npy", np.concatenate([oR, oR[-1] + np.cumsum(lens)]))
for f in ("Y", "EL"):
    a = np.load(f"{R}/{f}.npy", mmap_mode="r"); b = np.load(f"{S}/{f}.npy", mmap_mode="r")
    np.save(f"{D}/{f}.npy", np.concatenate([np.asarray(a), np.asarray(b[rows])]))
for f in ("cond", "date", "panel", "t", "is_eval", "is_sel"):
    a = np.load(f"{R}/{f}.npy", allow_pickle=True); b = np.load(f"{S}/{f}.npy", allow_pickle=True)[ks]
    if f in ("is_eval", "is_sel"):
        b = np.zeros(len(ks), bool)
    np.save(f"{D}/{f}.npy", np.concatenate([a, b.astype(a.dtype)]))
np.save(f"{D}/names.npy", np.array(nr))
x = np.load(f"{D}/X.npy", mmap_mode="r"); o = np.load(f"{D}/off.npy")
assert o[-1] == len(x) and len(o) - 1 == len(np.load(f"{D}/cond.npy")), "offsets"
assert np.array_equal(np.asarray(x[:nR:997]), np.asarray(XR[::997]), equal_nan=True), "real rows changed"
print("done", x.shape, "windows", len(o) - 1)
