"""B13d (10/29, lane 11): is a jam's end a deterministic function of the demand arriving at its tail?
Training-month truth. For events (B13, >= 1 h, birth row raw-observed) at the known sites (b13_sites), per row of the jam:
u = mean flow per lane over the 3 links upstream of the tail (free flow, observed only), and whether the jam ends within
the next 6 rows. AUC of -u, of -u / (site's flow per lane at birth), and of the age, for 'ends within 30 min'.
Also the day factor: per (panel, day), mean start-time anomaly over sites vs mean end-time anomaly."""
import sys, numpy as np, pandas as pd
sys.path.insert(0, r"C:/Users/user/tfb/src")
from cachelib import load
from common import PANELS, WORK
from sklearn.metrics import roc_auc_score
D = pd.read_parquet(WORK + "/b13_events.parquet"); S = pd.read_parquet(WORK + "/b13_sites.parquet")
K = set(zip(S.panel, S["head"])); D = D[D.raw0 & (D.dur >= 12) & np.array([k in K for k in zip(D.panel, D["head"])])]
Y, U, R, A = [], [], [], []
for p, d in D.groupby("panel"):
    P = load(p, "train"); ql = P.Q / P.lanes[None, :]
    for e in d.itertuples():
        tl = e.tail0; ups = list(range(max(0, tl - 3), tl))
        if not ups:
            continue
        u0 = np.nanmean(ql[max(0, e.t0 - 3):e.t0, ups])
        for r in range(e.t0 + 1, e.t0 + e.dur):
            u = np.nanmean(ql[r, ups]) if np.isfinite(ql[r, ups]).any() else np.nan
            if np.isfinite(u) and np.isfinite(u0) and u0 > 0:
                Y.append(e.t0 + e.dur - r <= 6); U.append(u); R.append(u / u0); A.append(r - e.t0)
    load.cache_clear()
Y, U, R, A = map(np.array, (Y, U, R, A))
print(f"rows {len(Y)} | ends within 30 min {Y.mean():.3f} | AUC: -u {roc_auc_score(Y, -U):.3f} | -u/u_birth {roc_auc_score(Y, -R):.3f} | age {roc_auc_score(Y, A):.3f}")
D = D.assign(end=D.tod + D.dur); D["sa"] = D.tod - D.groupby(["panel", "head"]).tod.transform("median")
D["ea"] = D.end - D.groupby(["panel", "head"]).end.transform("median")
G = D.groupby(["panel", "day"]).agg(sa=("sa", "mean"), ea=("ea", "mean"), n=("sa", "size")); G = G[G.n >= 3]
print(f"(panel, day) with >= 3 site events: {len(G)} | corr(mean start anomaly, mean end anomaly) {np.corrcoef(G.sa, G.ea)[0, 1]:+.2f}")
# leave-one-site-out: does the day's start anomaly at OTHER sites predict this event's end anomaly?
D = D.merge(G[["sa", "n"]].rename(columns={"sa": "sa_day", "n": "n_day"}), left_on=["panel", "day"], right_index=True)
D["sa_oth"] = (D.sa_day * D.n_day - D.sa) / (D.n_day - 1)
print(f"events {len(D)} | corr(other sites' start anomaly, this end anomaly) {np.corrcoef(D.sa_oth, D.ea)[0, 1]:+.2f} | corr(own start anomaly, own end anomaly) {np.corrcoef(D.sa, D.ea)[0, 1]:+.2f}")
