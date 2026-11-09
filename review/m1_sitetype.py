"""M1: bottleneck site types from the train episodes and where the far predictions sit (train precision, test share).
site = head link of episodes (>= 30 min) with >= 5 episodes at head+-1; type by the median discharge ratio r_cap:
  'capacity' r >= 0.9 (discharge = fd capacity: lane drop, demand visible on the mainline)
  'hidden'   r < 0.8  (discharge below capacity: off-ramp / diverge spillback, driven by unobserved exit demand)
a far predicted link j is assigned to the nearest site head h with 0 <= h - j <= 5 (the queue backs up from h)."""
import numpy as np, pandas as pd
E = pd.read_pickle("m1_episodes.pkl"); E = E[(E.dur >= 6) & E.r_cap.notna()]
F = pd.read_pickle("m1_farcells.pkl"); W = pd.read_pickle("m1_windows.pkl")
import sys; sys.path.insert(0, r"C:/Users/user/tfb/src")
from cachelib import T2_PANELS
sites = {}
for p, g in E.groupby("panel"):
    S = []
    for h in sorted(g.head0.unique()):
        m = g[(g.head0 - h).abs() <= 1]
        if len(m) >= 5:
            S.append((h, m.r_cap.median(), len(m)))
    sites[p] = S
for p in sites:
    s = sites[p]; print(p, "sites", len(s), "capacity", sum(r >= 0.9 for _, r, _ in s), "hidden", sum(r < 0.8 for _, r, _ in s),
                       "episodes at hidden", sum(n for _, r, n in s if r < 0.8), "of", sum(n for _, _, n in s))
kp = dict(zip(W.k, W.panel))
dt = "E:/tfb_nn/nn_test_afdem/"; wid = np.load(dt + "window_id.npy").astype(str)
def ptype(row):
    p = kp[row.k] if row.split == "train" else "_".join(wid[row.k].split("_")[1:4])
    best = None
    for h, r, n in sites.get(p, []):
        if 0 <= h - row.j <= 5 and (best is None or h - row.j < best[0]):
            best = (h - row.j, r)
    if best is None: return "none"
    return "capacity" if best[1] >= 0.9 else ("hidden" if best[1] < 0.8 else "mid")
F["stype"] = [ptype(r) for r in F.itertuples()]
a = F.groupby(["split", "stype"]).agg(cells=("npred", "sum"), tp=("tp", "sum")).reset_index()
a["share"] = a.cells / a.groupby("split").cells.transform("sum"); a["precision"] = a.tp / a.cells
print(a.round(3).to_string())
F["panel"] = [kp[r.k] if r.split == "train" else "_".join(wid[r.k].split("_")[1:4]) for r in F.itertuples()]
F["test"] = F.split != "train"
F["hid"] = F.npred * (F.stype == "hidden")
b = F.groupby(["panel", "test"]).agg(cells=("npred", "sum"), hid=("hid", "sum"), tp=("tp", "sum")); b["hidden"] = b.hid / b.cells; b["prec"] = b.tp / b.cells
b = b[["cells", "hidden", "prec"]].unstack()
print(b.round(3).to_string())
tr = F[~F.test]; w_test = F[F.test].groupby("panel").npred.sum() / F[F.test].npred.sum()
h_tr = tr.groupby("panel").apply(lambda g: (g.npred * (g.stype == "hidden")).sum() / g.npred.sum(), include_groups=False)
print("train hidden share re-weighted to the test panel mix:", round(float((h_tr * w_test).sum()), 3))
