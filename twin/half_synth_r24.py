"""R24 (all 24 worlds kept) with every other synthetic window (GPU memory: 27M rows did not fit), no domain token;
was: the domain-token data with HALF of the synthetic windows (every other one), real windows unchanged.
nn_data_afdemsimRd -> nn_data_afdemsimRd50 (81 columns)."""
import os, shutil, numpy as np
src, dst = "E:/tfb_nn/nn_data_afdemsimR24", "E:/tfb_nn/nn_data_afdemsimR24h"; os.makedirs(dst, exist_ok=True)
X = np.load(f"{src}/X.npy", mmap_mode="r"); off = np.load(f"{src}/off.npy"); date = np.load(f"{src}/date.npy", allow_pickle=True)
syn = np.array([str(d)[:4] >= "2040" for d in date]); keep = ~syn.copy(); keep[np.flatnonzero(syn)[::2]] = True
ks = np.flatnonzero(keep); print("windows kept", len(ks), "synthetic", int((keep & syn).sum()), flush=True)
lens = off[ks + 1] - off[ks]; noff = np.concatenate([[0], np.cumsum(lens)])
out = np.lib.format.open_memmap(f"{dst}/X.npy", mode="w+", dtype=np.float32, shape=(int(noff[-1]), int(X.shape[1])))
for j, k in enumerate(ks):
    out[noff[j]:noff[j + 1]] = X[off[k]:off[k + 1]]
out.flush(); del out
np.save(f"{dst}/off.npy", noff)
rows = np.concatenate([np.arange(off[k], off[k + 1]) for k in ks])
for f in ("Y", "EL"):
    np.save(f"{dst}/{f}.npy", np.asarray(np.load(f"{src}/{f}.npy", mmap_mode="r")[rows]))
for f in ("cond", "date", "panel", "t", "is_eval", "is_sel"):
    np.save(f"{dst}/{f}.npy", np.load(f"{src}/{f}.npy", allow_pickle=True)[ks])
shutil.copyfile(f"{src}/names.npy", f"{dst}/names.npy")
assert (np.load(f"{dst}/is_sel.npy") == np.load(f"{src}/is_sel.npy")[ks]).all()
print("done", noff[-1], "rows", flush=True)
