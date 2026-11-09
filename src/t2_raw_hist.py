"""X3 step 1 (10/23, strategy: X3 = a spatio-temporal net for ongoing diversity, time-boxed to 10/31): the RAW history of
each window's links, rows T-12..T (13 rows; row T is empty under the Task 2 rule, NO_T), exactly as the engineered features
see it (t2_model.PanelInputs.window: the same arrays window_X / arrival / demand read), 3 channels: v / vcut, q / cap,
observed (0/1); missing -> 0. Row order = the dataset's rows (window by window, the panel's links in order).
Datasets: train layout (panel.npy, t.npy; nn_data_afdem = the train months, nn_data_march = March) or test layout
(window_id.npy: official Q2_<panel>_<split>_<n> or M3 M3_<panel>_<t>).
usage: python t2_raw_hist.py <dataset dir under NN> [...]   -> NN/<dir>/R.npy (rows, 13, 3) float16"""
import os, sys, time, warnings, numpy as np, pandas as pd
os.environ["TFB_T2_NOT"] = "1"; os.environ["TFB_T2_ALLFIN"] = "1"; warnings.filterwarnings("ignore")
from cachelib import load, T2_PANELS
from common import REL, NN
from t2_model import PanelInputs

H = 12


def raw(PI, P, t):
    V, Q, tl = PI.window(t)
    v = V[tl - H:tl + 1] if tl >= H else np.vstack([np.full((H - tl, P.n), np.nan), V[:tl + 1]])   # first rows of a day:
    q = Q[tl - H:tl + 1] if tl >= H else np.vstack([np.full((H - tl, P.n), np.nan), Q[:tl + 1]])   # earlier rows absent
    ob = np.isfinite(v)
    return np.stack([np.where(ob, v / P.vcut[None, :], 0.0), np.where(np.isfinite(q), q / P.cap[None, :], 0.0), ob.astype(float)], axis=-1).transpose(1, 0, 2)


def windows(d):
    """[(panel, split, t)] in the dataset's window order"""
    if os.path.exists(f"{NN}/{d}/panel.npy"):
        pan = np.load(f"{NN}/{d}/panel.npy", allow_pickle=True); tt = np.load(f"{NN}/{d}/t.npy")
        sp = "validation" if d.startswith("nn_data_march") else "train"
        return [(str(p), sp, int(t)) for p, t in zip(pan, tt)]
    wid = [str(w) for w in np.load(f"{NN}/{d}/window_id.npy", allow_pickle=True)]; out = []
    if wid[0].startswith("M3_"):
        for w in wid:
            p, t = w[3:w.rindex("_")], int(w[w.rindex("_") + 1:]); out.append((p, "validation", t))
        return out
    idx = {}
    for sp in ("validation", "private"):
        for p in T2_PANELS:
            ti = {s: i for i, s in enumerate(load(p, sp).times)}
            for r in pd.read_csv(REL / "task2" / p / sp / "window_index.csv").itertuples():
                idx[r.window_id] = (p, sp, ti[r.forecast_origin])
            load.cache_clear()
    return [idx[w] for w in wid]


for d in (sys.argv[1:] if __name__ == "__main__" else []):
    t0 = time.time(); W_ = windows(d); off = np.load(f"{NN}/{d}/off.npy"); assert len(off) == len(W_) + 1, (d, len(off), len(W_))
    R = np.lib.format.open_memmap(f"{NN}/{d}/R.npy", mode="w+", dtype=np.float16, shape=(int(off[-1]), 13, 3))
    order = sorted(range(len(W_)), key=lambda i: (W_[i][0], W_[i][1]))       # one panel month in memory at a time
    cur = None
    for i in order:
        p, sp, t = W_[i]
        if cur != (p, sp):
            load.cache_clear(); P = load(p, sp); PI = PanelInputs(P); cur = (p, sp)
        r = raw(PI, P, t); assert len(r) == off[i + 1] - off[i], (d, i, len(r), off[i + 1] - off[i])
        R[off[i]:off[i + 1]] = r.astype(np.float16)
    R.flush(); del R
    print(d, len(W_), "windows", int(off[-1]), "rows", f"{time.time() - t0:.0f}s", flush=True)
