"""G340 report (#271): ys per window (as G337 / G338, from R) joined to G340's per-window oracle rows; per instrument and ys stratum: base
IoU, each oracle's gain and its share of the stratum's headroom (1 - base); the stratum's share of the instrument's total loss."""
import numpy as np, pandas as pd
W = "C:/Users/user/tfb/work/t2/"; NN = "C:/tfb_data/tfb_nn/"


def ys_of(Rw):
    o = Rw[:, :12, 2].astype(float) > 0; v = Rw[:, :12, 0].astype(float); q = o & (v <= 1.0); rows = np.flatnonzero(o.any(axis=0))
    if not len(rows):
        return True
    ql = q[:, rows[-1]]; n = int(ql.sum()); age = 0
    for r in rows[::-1]:
        if (q[:, r] & ql).any():
            age += 1
        else:
            break
    return n <= 3 or age <= 3


RM = np.load(NN + "nn_test_m3v3/R.npy", mmap_mode="r"); om = np.load(NN + "nn_test_m3v3/off.npy"); pm = {str(w): i for i, w in enumerate(np.load(NN + "nn_test_m3v3/window_id.npy", allow_pickle=True))}
RA = np.load(NN + "nn_data_afdem/R.npy", mmap_mode="r"); oa = np.load(NN + "nn_data_afdem/off.npy")
OG = ("O_box", "O_tail", "O_new", "O_box+tail")
for lab, f, src in (("M3", "M3_dev_ongoing", "m"), ("JF", "January_February_ongoing", "a"), ("JUN", "June_2030_ongoing", "a")):
    D = pd.read_csv(W + f"g340_{f}.csv")
    D["ys"] = [ys_of(np.asarray(RM[om[pm[str(w)]]:om[pm[str(w)] + 1]])) if src == "m" else ys_of(np.asarray(RA[oa[int(w)]:oa[int(w) + 1]])) for w in D.w]
    tot = (1 - D.base).sum()
    for g, d in D.groupby("ys"):
        hr = (1 - d.base).mean()
        print(f"{lab} ys={g}: n {len(d)}, base {d.base.mean():.4f}, share of the total loss {(1 - d.base).sum() / tot:.2f} | " + ", ".join(f"{o} {(d[o] - d.base).mean():+.4f} ({(d[o] - d.base).mean() / max(hr, 1e-9):.0%})" for o in OG), flush=True)
