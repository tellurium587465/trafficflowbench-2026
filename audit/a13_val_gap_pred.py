"""Audit A13: what the submitted Task 1 (s17 = state_f3ens3c) predicts inside the validation gaps.
Congested share (v < 0.6 vf) of predicted gap targets vs the true congested share inside the official TRAIN gaps
(onset 3.4%, ongoing 16.5%; a4), and vs the T2 forecast of the same cells (s17 queue file)."""
import sys, numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, r"C:/Users/user/tfb/src")
from cachelib import load, T2_PANELS
from common import REL

st = pd.read_parquet("C:/Users/user/tfb/work/t1/state_f3ens3c.parquet", columns=["panel", "timestamp", "link_id", "speed_kmh"])
qu = pd.read_parquet("C:/Users/user/tfb/work/t2/queue_s17_t1f3ens3c_t2v12nppdem_nnDnp6_t4l2.parquet")
rows = []
for p in T2_PANELS:
    P = load(p, "validation"); ti = {s: i for i, s in enumerate(P.times)}; li = {l: i for i, l in enumerate(P.links)}
    s = st[st.panel == p]; r = s.timestamp.map(ti); keep = r.notna(); s = s[keep]
    Vp = np.full((P.T, P.n), np.nan); Vp[r[keep].astype(int).to_numpy(), s.link_id.map(li).to_numpy()] = s.speed_kmh.to_numpy()
    wi = pd.read_csv(REL / "task2" / p / "validation" / "window_index.csv")
    for w in wi.itertuples():
        t = ti[w.forecast_origin]
        g = Vp[t + 1:t + 7]; m = np.isfinite(g)                                   # horizon part of the gap
        cong_pred = (g < 0.6 * P.vf[None, :]) & m
        qw = qu[qu.window_id == w.window_id]
        qp = np.zeros((6, P.n), bool)
        rr = qw.timestamp.map(ti).to_numpy() - t - 1; cc = qw.link_id.map(li).to_numpy()
        qp[rr, cc] = qw.queue_pred.to_numpy().astype(bool)
        rows.append(dict(panel=p, cond=w.condition, n=int(m.sum()), t1_cong=int(cong_pred.sum()), t2_queue_at_t1_targets=int((qp & m).sum())))
    load.cache_clear()
d = pd.DataFrame(rows).groupby("cond")[["n", "t1_cong", "t2_queue_at_t1_targets"]].sum()
d["t1_cong_share"] = d.t1_cong / d.n; d["t2_queue_share"] = d.t2_queue_at_t1_targets / d.n
print(d.round(4))
