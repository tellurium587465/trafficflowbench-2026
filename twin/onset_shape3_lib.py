"""raw C: matrices (speed, flow, eligibility) for a cached panel's links and times (the caches live on the E: HDD)."""
import glob, numpy as np, pandas as pd
from common import REL


def raw_matrices(P):
    li = {l: i for i, l in enumerate(P.links)}; ti = {t: i for i, t in enumerate(P.times)}
    V = np.full((len(P.times), P.n), np.nan, np.float32); Q = V.copy(); E = np.zeros((len(P.times), P.n), bool)
    for f in sorted(glob.glob(str(REL / "corridors" / P.panel / "train" / "mainline_states" / "**" / "*.parquet"), recursive=True)):
        d = pd.read_parquet(f, columns=["timestamp", "link_id", "speed_kmh", "flow_vph", "is_score_eligible"])
        g = d.groupby(["timestamp", "link_id"], sort=False).agg(v=("speed_kmh", "mean"), q=("flow_vph", "mean"), e=("is_score_eligible", "max")).reset_index()
        r = g.timestamp.map(ti).to_numpy(); c = g.link_id.map(li).to_numpy(); ok = ~(pd.isna(r) | pd.isna(c))
        r, c = r[ok].astype(int), c[ok].astype(int)
        V[r, c] = g.v.to_numpy()[ok]; Q[r, c] = g.q.to_numpy()[ok]; E[r, c] = g.e.to_numpy()[ok] > 0
    return V, Q, E


def blocks(V, P, rows=None):
    """the generator's blocks (block_structure.py): adjacent pairs congested >= 30 rows with split rate <= 0.05 are joined"""
    V = V if rows is None else V[rows]
    ok = np.isfinite(V); cong = (V <= P.vcut[None, :]) & ok
    a, b = cong[:, :-1], cong[:, 1:]; both = ok[:, :-1] & ok[:, 1:]
    n_any = ((a | b) & both).sum(0); split = ((a ^ b) & both).sum(0) / np.maximum(n_any, 1)
    tight = (n_any >= 30) & (split <= 0.05); bid = np.zeros(P.n, int)
    for i in range(1, P.n):
        bid[i] = bid[i - 1] if tight[i - 1] else bid[i - 1] + 1
    return bid


def split_cells(P, split):
    """visible cells of a masked split (validation / private): link index, speed, flow (mean over stations)"""
    li = {str(l): i for i, l in enumerate(P.links)}; rows = []
    for f in sorted(glob.glob(str(REL / "corridors" / P.panel / split / "mainline_states_masked" / "**" / "*.parquet"), recursive=True)):
        d = pd.read_parquet(f, columns=["timestamp", "link_id", "speed_kmh", "flow_vph"]).dropna()
        g = d.groupby(["timestamp", "link_id"], sort=False).agg(v=("speed_kmh", "mean"), q=("flow_vph", "mean")).reset_index()
        g["c"] = g.link_id.astype(str).map(li); rows.append(g.dropna(subset=["c"]))
    g = pd.concat(rows); return g.c.to_numpy(int), g.v.to_numpy(float), g.q.to_numpy(float)
