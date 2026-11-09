"""B13f (10/29, lane 11): how much does the time of day tell WHERE a jam is born? Training months, births at the known sites
(B13 events, raw0). For each birth at row tau in month m: the per-site birth rate at tod(tau) +-2 rows from the other
months (per day at risk), and the per-site birth count without time; rank of the true site among the panel's known sites
free at tau. Top-1 share and mean reciprocal rank."""
import sys, numpy as np, pandas as pd
sys.path.insert(0, r"C:/Users/user/tfb/src")
from common import WORK
D = pd.read_parquet(WORK + "/b13_events.parquet"); S = pd.read_parquet(WORK + "/b13_sites.parquet")
K = set(zip(S.panel, S["head"])); D = D[np.array([k in K for k in zip(D.panel, D["head"])])].copy(); D["ym"] = D.day.str[:7]
res = {"tod": [], "count": []}
for p, d in D.groupby("panel"):
    sites = sorted({h for (pp, h) in K if pp == p})
    if len(sites) < 2:
        continue
    for ym in d.ym.unique():
        tr, te = d[d.ym != ym], d[(d.ym == ym) & d.raw0]
        cnt = tr.groupby("head").size().reindex(sites, fill_value=0).to_numpy(float)
        H = np.zeros((len(sites), 288))
        for i, s in enumerate(sites):
            h = np.bincount(tr[tr["head"] == s].tod.to_numpy() % 288, minlength=288).astype(float)
            H[i] = sum(np.roll(h, r) for r in range(-2, 3))
        for e in te.itertuples():
            busy = set(d[(d.ym == ym) & (d.t0 < e.t0) & (d.t0 + d.dur > e.t0)]["head"])     # sites already jammed at tau
            free = [i for i, s in enumerate(sites) if s not in busy or s == e.head]
            j = sites.index(e.head)
            for nm, sc in (("tod", H[:, e.tod % 288]), ("count", cnt)):
                r = 1 + sum(sc[i] > sc[j] for i in free if i != j) + 0.5 * sum(sc[i] == sc[j] for i in free if i != j)
                res[nm].append((r, len(free)))
for nm, v in res.items():
    v = np.array(v); print(f"{nm:6s} births {len(v)} | top-1 {np.mean(v[:, 0] <= 1):.3f} | MRR {np.mean(1 / v[:, 0]):.3f} | free sites (mean) {v[:, 1].mean():.1f}")
