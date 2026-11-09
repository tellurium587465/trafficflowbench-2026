"""G51: E4 proxy (STATE.md 10/22 pre-registration, strategy session 10/22). Month-CV fold 2 OOF (FOLDS=2) of the
attnglob configuration, seeds 0,1, three arms: none (e4p_none: the added shifted month W1 = world C6 held out),
x1 (e4p_x1: W1 trained with weight 1), x4 (e4p_x4: SIMW=4). Primary: the held-out shifted worlds C7, C8, C16, C17
(templates Jan / Feb 2031, unseen months), ongoing IoU over the six steps; onset IoU at T+30 (as the M3 gate) is
reference only. Secondary: the real selector-like windows of Jan-Feb 2031 (onset restricted to mornings as G50).
Seeds averaged as probabilities before decoding; paired differences, cluster bootstrap (panel x day).
usage: python g51_e4_proxy.py"""
import sys, glob, os, pickle, numpy as np, pandas as pd
sys.path.insert(0, r"C:/Users/user/tfb/src")
from common import NN
from t2_model import best_set

W = "C:/Users/user/tfb/work/t2/"
D = f"{NN}/nn_data_e4p1"
ARMS = ("e4p_none", "e4p_x1", "e4p_x4")
EVAL = ("2223-10", "2224-05", "2229-07", "2229-08")
off = np.load(f"{D}/off.npy"); Y = np.load(f"{D}/Y.npy", mmap_mode="r"); EL = np.load(f"{D}/EL.npy", mmap_mode="r")
cond = np.load(f"{D}/cond.npy", allow_pickle=True); date = np.load(f"{D}/date.npy", allow_pickle=True)
panel = np.load(f"{D}/panel.npy", allow_pickle=True); tt = np.load(f"{D}/t.npy", allow_pickle=True)
is_sel = np.load(f"{D}/is_sel.npy").astype(bool)


def iou(p, y, e, c):
    if c == "queue_onset":                                  # decoded and scored at T+30 (M3 gate convention)
        p, y, e = p[-1:], y[-1:], e[-1:]
    s = best_set(p.ravel(), np.ones(p.size, bool)).reshape(p.shape)
    u = ((s | y) & e).sum(); return 1.0 if u == 0 else float((s & y & e).sum() / u)


O = {}
for a in ARMS:
    fs = sorted(glob.glob(W + f"oof_{a}_s[0-9].pkl")); assert len(fs) == 2, (a, fs)
    O[a] = [pickle.load(open(f, "rb")) for f in fs]
keys = sorted(set.intersection(*[set(o) for os_ in O.values() for o in os_]))
rows = []
for k in keys:
    d = str(date[k]); c = str(cond[k])
    if d[:7] in EVAL:
        part = "synthetic C7/C8/C16/C17"
    elif is_sel[k] and d[:7] in ("2031-01", "2031-02") and not (c == "queue_onset" and (int(tt[k]) + 6) % 288 >= 144):
        part = "real Jan-Feb 2031"
    else:
        continue
    y = np.asarray(Y[off[k]:off[k + 1]]).T.astype(bool); e = np.asarray(EL[off[k]:off[k + 1]]).T.astype(bool)
    r = dict(part=part, cond=c, cl=f"{panel[k]}_{d[:10]}")
    for a, os_ in O.items():
        r[a] = iou(np.mean([o[k] for o in os_], axis=0), y, e, c)
    rows.append(r)
R = pd.DataFrame(rows); R.to_csv(W + "g51_e4_proxy.csv", index=False); rng = np.random.default_rng(0)


def diff(g, a, b):
    u = g.cl.unique(); idx = {x: np.flatnonzero(g.cl.to_numpy() == x) for x in u}; d = (g[b] - g[a]).to_numpy()
    se = np.std([d[np.concatenate([idx[x] for x in rng.choice(u, len(u))])].mean() for _ in range(1000)])
    return d.mean(), se


for (part, c), g in R.groupby(["part", "cond"]):
    print(f"{part} {c}: windows {len(g)}, " + ", ".join(f"{a} {g[a].mean():.4f}" for a in ARMS))
    for a, b in (("e4p_none", "e4p_x1"), ("e4p_none", "e4p_x4"), ("e4p_x1", "e4p_x4")):
        m, se = diff(g, a, b); print(f"   {b} - {a}: {m:+.4f} (SE {se:.4f}, {m / max(se, 1e-9):+.1f} SE)", flush=True)
