"""Append the demand-profile columns (demcache) to the NN training matrix -> E:/tfb_nn/nn_data_dem (streamed)."""
import os, shutil, numpy as np
from t2_demand import NAMES

src, dst = "E:/tfb_nn/nn_data", "E:/tfb_nn/nn_data_dem"
os.makedirs(dst, exist_ok=True)
X = np.load(f"{src}/X.npy", mmap_mode="r")
DEM = np.load("C:/Users/user/tfb/work/t2/demcache2_noT.npy", mmap_mode="r")
assert len(X) == len(DEM), (X.shape, DEM.shape)
out = np.lib.format.open_memmap(f"{dst}/X.npy", mode="w+", dtype=np.float32, shape=(len(X), X.shape[1] + DEM.shape[1]))
for a in range(0, len(X), 1 << 21):
    out[a:a + (1 << 21), :X.shape[1]] = X[a:a + (1 << 21)]
    out[a:a + (1 << 21), X.shape[1]:] = DEM[a:a + (1 << 21)]
out.flush()
np.save(f"{dst}/names.npy", np.array(list(np.load(f"{src}/names.npy", allow_pickle=True)) + NAMES))
for f in os.listdir(src):
    if f not in ("X.npy", "names.npy"):
        shutil.copyfile(f"{src}/{f}", f"{dst}/{f}")
print("ok", out.shape, sorted(os.listdir(dst)))
