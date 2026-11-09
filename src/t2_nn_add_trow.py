"""Append the origin row T to the af+dem NN inputs (10/12, G27): the organizer ruling (discussion 742068) allows any
released data with a timestamp at or before T, and the T row was excluded only as a conservative reading (NO_T).
In the test months the T row comes from the Task 1-masked layer (48-67% observed); on train the same layer is the
released masked layer = finite and not a Task 1 blank. Persistence from the masked T row beats persistence from T-5
on the official-like ongoing windows: 0.685 vs 0.591.

4 columns per link: trow_obs (observed flag), trow_vr (v/vf, 0 if unobserved), trow_qc (q/cap, 0 if unobserved),
trow_queue (v <= v_cut, 0 if unobserved). The 80 existing columns are copied unchanged, so ZERO=...+trow_* restores
the old model exactly (identity check below).
nn_data_afdem -> nn_data_afdemtrow (train rows in featcache order), nn_test_afdem -> nn_test_afdemtrow.
"""
import os, shutil, time, numpy as np, pandas as pd
from cachelib import load
from common import REL

NAMES = ["trow_obs", "trow_vr", "trow_qc", "trow_queue"]
t0 = time.time(); PC = {}


def panel(p, split):
    if (p, split) not in PC:
        P = load(p, split)
        obs = np.isfinite(P.V) & np.isfinite(P.Q)
        if split == "train":
            obs &= ~P.blank                                     # the released masked layer of the train split
        PC[(p, split)] = (P, obs)
    return PC[(p, split)]


def feats(p, split, t):
    P, obs = panel(p, split); o = obs[t]
    vr = np.where(o, P.V[t] / P.vf, 0.0); qc = np.where(o, P.Q[t] / P.cap, 0.0)
    qu = np.where(o, P.V[t] <= P.vcut, False)
    return np.column_stack([o, vr, qc, qu]).astype(np.float32)


for src, dst in (("E:/tfb_nn/nn_data_afdem", "E:/tfb_nn/nn_data_afdemtrow"), ("E:/tfb_nn/nn_test_afdem", "E:/tfb_nn/nn_test_afdemtrow")):
    os.makedirs(dst, exist_ok=True)
    X = np.load(f"{src}/X.npy", mmap_mode="r"); offs = np.load(f"{src}/off.npy")
    TR = np.zeros((len(X), len(NAMES)), np.float32)
    if "nn_data" in src:
        pan = np.load(f"{src}/panel.npy", allow_pickle=True); tt = np.load(f"{src}/t.npy")
        for k in range(len(offs) - 1):
            f = feats(str(pan[k]), "train", int(tt[k])); assert len(f) == offs[k + 1] - offs[k], ("links", k)
            TR[offs[k]:offs[k + 1]] = f
            if k % 10000 == 0:
                print("train", k, f"{time.time()-t0:.0f}s", flush=True)
    else:
        wid = np.load(f"{src}/window_id.npy", allow_pickle=True).astype(str); TI = {}
        for i, w in enumerate(wid):
            split = "validation" if "_validation_" in w else "private"; p = w[3:w.index("_" + split + "_")]
            if (p, split) not in TI:
                P, _ = panel(p, split); ti = {s: j for j, s in enumerate(P.times)}
                wi = pd.read_csv(REL / "task2" / p / split / "window_index.csv")
                TI[(p, split)] = {r.window_id: ti[r.forecast_origin] for r in wi.itertuples()}
            f = feats(p, split, TI[(p, split)][w]); assert len(f) == offs[i + 1] - offs[i], ("links", w)
            TR[offs[i]:offs[i + 1]] = f
    print(dst, "observed share", round(float(TR[:, 0].mean()), 3), "queued share among observed",
          round(float(TR[TR[:, 0] > 0, 3].mean()), 4), flush=True)
    out = np.lib.format.open_memmap(f"{dst}/X.npy", mode="w+", dtype=np.float32, shape=(len(X), X.shape[1] + len(NAMES)))
    for a in range(0, len(X), 1 << 21):
        out[a:a + (1 << 21), :X.shape[1]] = X[a:a + (1 << 21)]
        out[a:a + (1 << 21), X.shape[1]:] = TR[a:a + (1 << 21)]
    out.flush()
    assert np.array_equal(np.asarray(out[::997, :X.shape[1]]), np.asarray(X[::997]), equal_nan=True), "existing columns changed"
    del out
    for f in os.listdir(src):
        if f == "names.npy":
            np.save(f"{dst}/names.npy", np.array(list(np.load(f"{src}/names.npy", allow_pickle=True)) + NAMES))
        elif f != "X.npy":
            shutil.copyfile(f"{src}/{f}", f"{dst}/{f}")
    print("done", dst, f"{time.time()-t0:.0f}s", flush=True)
