"""Corridor-relative hazard features (aligned with featcache rows).

Onsets away from the recurrent bottlenecks are not random: they sit where arriving demand / downstream minimum
capacity is high *within the corridor at that moment* (median percentile 0.88, same as recurrent onsets).
A per-link GBDT cannot compare links, so expose each link's standing in its window: percentile, fraction of
the window maximum, signed/absolute distance to the window argmax and a local-maximum flag.
"""
import os, time, numpy as np
C = os.environ.get("TFB_T2CACHE", "2_noT")
Z = np.load(rf"C:/Users/user/tfb/work/t2/featcache{C}.npz", allow_pickle=True)
off = Z["off"]; names = list(Z["names"]); X = Z["X"]
ARR = np.load(rf"C:/Users/user/tfb/work/t2/arrcache{C}.npy", mmap_mode="r")
iq, icap = names.index("last_q"), names.index("capl")
SRC = {"a30d": 6, "a20": 1, "a30": 2}
NAMES = []
for s in list(SRC) + ["qc"]:
    NAMES += [f"rel_{s}_pct", f"rel_{s}_frac"]
NAMES += ["rel_dmax", "rel_dmax_abs", "rel_lmax5", "rel_lmax20"]


def pct(v):
    o = np.argsort(v, kind="stable"); r = np.empty(len(v)); r[o] = np.arange(len(v)); return r / max(len(v) - 1, 1)


def lmax(v, h):
    n = len(v); p = np.pad(v, h, constant_values=-np.inf)
    m = np.max(np.lib.stride_tricks.sliding_window_view(p, 2 * h + 1), axis=1)[:n]
    return (v >= m).astype(np.float32)


out = np.zeros((off[-1], len(NAMES)), np.float32); t0 = time.time()
for k in range(len(off) - 1):
    a, b = off[k], off[k + 1]; n = b - a
    cols = {s: np.nan_to_num(np.asarray(ARR[a:b, j]), nan=-1.0) for s, j in SRC.items()}
    cols["qc"] = np.nan_to_num(X[a:b, iq] / np.maximum(X[a:b, icap], 1e-6), nan=-1.0)
    F = []
    for s, v in cols.items():
        mx = v.max(); F += [pct(v), v / mx if mx > 0 else np.zeros(n)]
    h = cols["a30d"]; am = int(np.argmax(h)); d = np.arange(n) - am
    F += [d.astype(np.float32), np.abs(d).astype(np.float32), lmax(h, 5), lmax(h, 20)]
    out[a:b] = np.column_stack(F)
    if k % 20000 == 0:
        print(k, f"{time.time()-t0:.0f}s", flush=True)
np.save(rf"C:/Users/user/tfb/work/t2/relcache{C}.npy", out)
print("done", out.shape, NAMES, f"{time.time()-t0:.0f}s")
