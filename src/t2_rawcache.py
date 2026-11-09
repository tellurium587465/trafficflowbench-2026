"""Raw 60-min history per link for the NN: v/vf and q/cap at T-60..T-5 (12 steps each, NaN where absent).
train -> rawcache2_noT.npy aligned with featcache rows; validation/private -> rawtest (nn_test window order).
Then nn_data_dem(+raw) / nn_test_dem(+raw) -> E:/tfb_nn/nn_data_demraw, nn_test_demraw."""
import os, shutil, warnings, numpy as np, pandas as pd
os.environ.setdefault("TFB_T2_NOT", "1"); warnings.filterwarnings("ignore")
from cachelib import load, T2_PANELS
from common import REL
from t2_model import PanelInputs

NAMES = [f"raw_v{i}" for i in range(12)] + [f"raw_q{i}" for i in range(12)]


def raw(PI, t):
    V, Q, tl = PI.window(t); P = PI.P
    rows = np.arange(tl - 12, tl)
    v = np.full((12, P.n), np.nan, np.float32); q = np.full((12, P.n), np.nan, np.float32)
    ok = rows >= 0
    v[ok] = V[rows[ok]] / P.vf[None, :]; q[ok] = Q[rows[ok]] / P.cap[None, :]
    return np.column_stack([v.T, q.T]).astype(np.float32)


Z = np.load("C:/Users/user/tfb/work/t2/featcache2_noT.npz", allow_pickle=True)
off, panel, tt = Z["off"], Z["panel"], Z["t"]
out = np.zeros((off[-1], 24), np.float32); PIs = {}
for k in range(len(off) - 1):
    p = str(panel[k])
    if p not in PIs:                                   # (setdefault would rebuild PanelInputs every call)
        PIs[p] = PanelInputs(load(p, "train"))
    PI = PIs[p]
    out[off[k]:off[k + 1]] = raw(PI, int(tt[k]))
rt = []
for split in ("validation", "private"):
    for p in T2_PANELS:
        P = load(p, split); PI = PanelInputs(P); ti = {s: i for i, s in enumerate(P.times)}
        for r in pd.read_csv(REL / "task2" / p / split / "window_index.csv").itertuples():
            rt.append(raw(PI, ti[r.forecast_origin]))
rt = np.vstack(rt)
for src, dst, A in (("E:/tfb_nn/nn_data_dem", "E:/tfb_nn/nn_data_demraw", out), ("E:/tfb_nn/nn_test_dem", "E:/tfb_nn/nn_test_demraw", rt)):
    os.makedirs(dst, exist_ok=True); X = np.load(f"{src}/X.npy", mmap_mode="r"); assert len(X) == len(A), (X.shape, A.shape)
    o = np.lib.format.open_memmap(f"{dst}/X.npy", mode="w+", dtype=np.float32, shape=(len(X), X.shape[1] + 24))
    for a in range(0, len(X), 1 << 21):
        o[a:a + (1 << 21), :X.shape[1]] = X[a:a + (1 << 21)]; o[a:a + (1 << 21), X.shape[1]:] = A[a:a + (1 << 21)]
    o.flush()
    for f in os.listdir(src):
        if f != "X.npy":
            shutil.copyfile(f"{src}/{f}", f"{dst}/{f}")
    if os.path.exists(f"{src}/names.npy"):
        np.save(f"{dst}/names.npy", np.array(list(np.load(f"{src}/names.npy", allow_pickle=True)) + NAMES))
    print("ok", dst, o.shape, flush=True)
