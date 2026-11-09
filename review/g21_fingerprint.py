"""G21 (B3): can the no-memory NN inputs identify the link (a fingerprint that lets the model memorise where queues
form)? Per panel, link rows of the selector-like windows; kNN (k=5, standardised) trained on months Jun-Nov, scored on
Dec-Feb; accuracy of the exact link index vs chance (1/n). Feature groups: all inputs the no-memory NN sees; demand
features only (per-link weekday x time-of-day profile and capacity); the rest (state + arrival + segment)."""
import numpy as np, pandas as pd
from sklearn.neighbors import KNeighborsClassifier
D = "E:/tfb_nn/nn_data_afdem"
names = list(np.load(f"{D}/names.npy", allow_pickle=True))
X = np.load(f"{D}/X.npy", mmap_mode="r"); off = np.load(f"{D}/off.npy"); panel = np.load(f"{D}/panel.npy", allow_pickle=True)
date = np.load(f"{D}/date.npy", allow_pickle=True); sel = np.load(f"{D}/is_sel.npy")
ZERO = {"pos", "tod", "dow", "capl", "lanes", "length", "vf", "w", "prior_on", "prior_og"}
DEM = {"dem10_c", "dem20_c", "dem30_c", "dem30_over_dnmin", "dem_shape30", "dem_level"}
ARR = {f"arr{i}" for i in range(7)}
groups = {"all no-memory inputs": [n for n in names if n not in ZERO],
          "demand features only": [n for n in names if n in DEM],
          "arrival-flow only": [n for n in names if n in ARR],
          "state/segment (no demand, no arrival)": [n for n in names if n not in ZERO | DEM | ARR]}
rng = np.random.default_rng(0)
for p in ("D7_I405_N", "D12_I5_N", "D7_I210_W"):
    ks = np.flatnonzero(sel & (panel == p)); mon = np.array([str(date[k])[:7] for k in ks])
    tr = ks[np.isin(mon, ["2030-06", "2030-07", "2030-08", "2030-09", "2030-10", "2030-11"])]; te = ks[np.isin(mon, ["2030-12", "2031-01", "2031-02"])]
    tr = rng.choice(tr, min(300, len(tr)), replace=False); te = rng.choice(te, min(120, len(te)), replace=False)
    def rows(idx):
        r = np.concatenate([np.arange(off[k], off[k + 1]) for k in idx]); y = np.concatenate([np.arange(off[k + 1] - off[k]) for k in idx])
        return np.asarray(X[r]), y
    Xa, ya = rows(tr); Xb, yb = rows(te); n = int(ya.max()) + 1
    out = []
    for g, cols in groups.items():
        j = [names.index(c) for c in cols]
        A, B = Xa[:, j].astype(np.float64), Xb[:, j].astype(np.float64)
        mu, sd = np.nanmean(A, 0), np.nanstd(A, 0) + 1e-9
        A = np.nan_to_num((A - mu) / sd); B = np.nan_to_num((B - mu) / sd)
        acc = (KNeighborsClassifier(5, n_jobs=8).fit(A, ya).predict(B) == yb).mean()
        out.append(f"{g}: {acc:.3f}")
    print(f"{p} (n={n}, chance {1/n:.3f}) | " + " | ".join(out), flush=True)
