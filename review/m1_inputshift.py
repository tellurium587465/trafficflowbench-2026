"""M1: input distribution of ongoing windows, train selector-like vs test (inputs only): per feature median / IQR and
the KS distance, separately for links near (<= 3) and far from the queue at T."""
import numpy as np, pandas as pd
names = list(np.load("E:/tfb_nn/nn_data_afdem/names.npy", allow_pickle=True)); ix = {c: i for i, c in enumerate(names)}
X = np.load("E:/tfb_nn/nn_data_afdem/X.npy", mmap_mode="r"); OFF = np.load("E:/tfb_nn/nn_data_afdem/off.npy")
cond = np.load("E:/tfb_nn/nn_data_afdem/cond.npy"); sel = np.load("E:/tfb_nn/nn_data_afdem/is_sel.npy")
dt = "E:/tfb_nn/nn_test_afdem/"; Xt = np.load(dt + "X.npy", mmap_mode="r"); offt = np.load(dt + "off.npy"); condt = np.load(dt + "cond.npy")
wid = np.load(dt + "window_id.npy").astype(str)
def rows(Xa, off, ks):
    out = []
    for k in ks:
        x = np.asarray(Xa[off[k]:off[k + 1]]); lq = x[:, ix["last_queue"]] == 1; near = np.zeros(len(x), bool)
        for j in np.flatnonzero(lq): near[max(0, j - 3):j + 4] = True
        out.append(np.column_stack([x, near]))
    return np.vstack(out)
tr = rows(X, OFF, np.flatnonzero(sel & (cond == "queue_ongoing")))
te = {s: rows(Xt, offt, [i for i in range(len(wid)) if condt[i] == "queue_ongoing" and s in wid[i]]) for s in ("validation", "private")}
F = ["tod", "last_v", "last_q", "v_m12", "q_m12", "q_today", "nq_hist", "n_queued_now", "dem_level", "dem30_c", "dem30_over_dnmin", "dem_shape30", "arr6", "s_flow", "head_rel"]
def ks(a, b):
    a = a[np.isfinite(a)]; b = b[np.isfinite(b)]
    if len(a) < 10 or len(b) < 10: return np.nan
    g = np.unique(np.concatenate([np.quantile(a, np.linspace(0, 1, 201)), np.quantile(b, np.linspace(0, 1, 201))]))
    return float(np.max(np.abs(np.searchsorted(np.sort(a), g, "right") / len(a) - np.searchsorted(np.sort(b), g, "right") / len(b))))
for reg, m in (("near", 1), ("far", 0)):
    print(f"--- {reg} links: train {int((tr[:, -1] == m).sum())} rows, val {int((te['validation'][:, -1] == m).sum())}, private {int((te['private'][:, -1] == m).sum())}")
    for c in F:
        a = tr[tr[:, -1] == m, ix[c]]; v = te["validation"][te["validation"][:, -1] == m, ix[c]]; p = te["private"][te["private"][:, -1] == m, ix[c]]
        q = lambda z: np.round(np.nanquantile(z, [0.25, 0.5, 0.75]), 3)
        print(f"{c:18s} train {q(a)} val {q(v)} priv {q(p)}  KS val {ks(a, v):.3f} priv {ks(a, p):.3f}")
