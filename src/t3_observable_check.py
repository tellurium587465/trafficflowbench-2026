"""Does the official 'observable transitions' filter explain real S_LWR (~0.47) << our proxy (~0.65)?

score_task3 keeps a sensor link only if all its topology neighbours (incl. latent mainline links) are present in
the submission frame. Compare the proxy S_LWR with/without that filter for a synthetic answer whose target-cell
error equals the observation noise (density x (1 + 2.4% white noise)), on the train split.
"""
import numpy as np, pandas as pd
from cachelib import load, PANELS
from common import REL
import metrics as M


def observable(P):
    topo = pd.read_csv(REL / "corridors" / P.panel / "network" / "lwr_mainline_topology.csv", dtype=str)
    ids = set(topo.link_id.astype(str)); sub = set(P.links)
    ok = {}
    for r in topo.itertuples(index=False):
        inc = {v.strip() for v in str(getattr(r, "incoming_link_ids", "")).split(";") if v.strip() and v.strip() != "nan"}
        out = {v.strip() for v in str(getattr(r, "outgoing_link_ids", "")).split(";") if v.strip() and v.strip() != "nan"}
        ok[str(r.link_id)] = (inc & ids).issubset(sub) and (out & ids).issubset(sub)
    return np.array([ok.get(l, False) for l in P.links]), len(ids)


rng = np.random.default_rng(0); rows = []
for p in PANELS:
    P = load(p, "train"); obs, nt = observable(P)
    noise = 1 + 0.024 * rng.standard_normal(P.V.shape)
    Vp = P.V.copy(); Qp = P.Q * noise                       # k error ~2.4% at target cells
    _, full = M.physics_score(P, Vp, Qp, detail=True)
    L0 = P.length.copy()
    P.length = np.where(obs, L0, 0.0)                       # zero length -> N = 0 -> link drops out of both sums
    _, filt = M.physics_score(P, Vp, Qp, detail=True)
    P.length = L0
    rows.append((p, P.n, nt, obs.mean(), np.mean([b for _, b in full]), np.mean([b for _, b in filt]),
                 np.median(L0), np.median(L0[obs]) if obs.any() else np.nan))
    print(f"{p:12s} sensor links {P.n:4d} topo links {nt:4d} observable {obs.mean():.2f} | S_LWR proxy all {rows[-1][4]:.3f}"
          f" observable-only {rows[-1][5]:.3f} | median length all {rows[-1][6]:.3f} obs {rows[-1][7]:.3f}", flush=True)
d = pd.DataFrame(rows, columns=["p", "n", "nt", "obs", "all", "filt", "lall", "lobs"])
print("mean S_LWR (noise-level answer): all links", round(d["all"].mean(), 3), "| observable only", round(d["filt"].mean(), 3))
