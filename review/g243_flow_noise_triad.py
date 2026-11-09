"""G243 (11/03, ledger #206, the strategy's (3), GOAL_FIRST_PLACE 8.5 pre-registered): the observation noise of flow by the
three-indicator (one-factor) method, against the adjacent-difference estimate behind the floor of 28 veh/h/lane.
Training months (train split = observations), consecutive mainline links (i, j, k) by order_index, all three in free flow
(v > 0.6 v_f) and finite at the same row, flow per lane, residual = minus the link's mean at the same 5-minute slot and day type
(weekday / weekend). Fluctuations shared along the triple are signal (one factor with loadings), noise is independent per cell:
  sigma2(j) = Var(r_j) - Cov(r_j, r_i) Cov(r_j, r_k) / Cov(r_i, r_k)      (middle link, main)
  sigma2(i) = Var(r_i) - Cov(r_i, r_j) Cov(r_i, r_k) / Cov(r_j, r_k)      (the strategy's form, first link)
  adjacent  = (Var(r_i - r_j) + Var(r_j - r_k)) / 4                       (the earlier method on the same samples)
Triples: "strict" = no flow enters or leaves between i and k (j and k have one incoming, i and j one outgoing: no ramps, no
connectors); "noramp" = no on / off ramp at j, k (incoming side) and i, j (outgoing side), connectors allowed. Day = 06-20 h.
Reads the training split only (no test month, no hold-out).
usage: py -B -X utf8 g243_flow_noise_triad.py"""
import sys
sys.path.insert(0, r"C:/Users/user/tfb/src")
import numpy as np, pandas as pd
from cachelib import load
from common import PANELS
rows = []
for p in PANELS:
    P = load(p, "train"); tp = P.topo.sort_values("order_index"); ix = [list(P.links).index(l) for l in tp.index]
    Q = P.Q[:, ix] / P.lanes[ix][None, :]; V = P.V[:, ix]; vf = P.vf[ix]; ok = np.isfinite(Q) & np.isfinite(V) & (V > 0.6 * vf[None, :])
    T_ = len(Q); slot = np.arange(T_) % 288; we = np.array([pd.Timestamp(P.times[d * 288][:10]).dayofweek >= 5 for d in range(T_ // 288)]).repeat(288)
    g = slot + 288 * we; Qm = np.where(ok, Q, np.nan); M = np.full((576, Q.shape[1]), np.nan)
    for gi in range(576):
        with np.errstate(all="ignore"):
            M[gi] = np.nanmean(Qm[g == gi], axis=0)
    Rr = Q - M[g]; day = (slot >= 72) & (slot < 240)
    nin = tp.n_incoming.to_numpy(); nout = tp.n_outgoing.to_numpy(); onr = tp.on_ramp_link_ids.notna().to_numpy(); offr = tp.off_ramp_link_ids.notna().to_numpy()
    for j in range(1, Q.shape[1] - 1):
        i, k = j - 1, j + 1
        strict = nin[j] == 1 and nin[k] == 1 and nout[i] == 1 and nout[j] == 1
        noramp = not (onr[j] or onr[k] or offr[i] or offr[j])
        if not noramp:
            continue
        m3 = ok[:, i] & ok[:, j] & ok[:, k]
        for per, pm in (("day", day), ("night", ~day)):
            mm = m3 & pm
            if mm.sum() < 1000:
                continue
            a, b, c = Rr[mm, i], Rr[mm, j], Rr[mm, k]; C = np.cov(np.vstack([a, b, c]))
            s2j = C[1, 1] - C[1, 0] * C[1, 2] / C[0, 2]; s2i = C[0, 0] - C[0, 1] * C[0, 2] / C[1, 2]; adj = (np.var(a - b) + np.var(b - c)) / 4
            rows.append(dict(panel=p, j=j, per=per, strict=strict, n=int(mm.sum()), q=float(np.nanmean(Q[mm, j])), sd_r=float(np.sqrt(C[1, 1])),
                             s2j=s2j, s2i=s2i, adj=adj, corr_ik=float(C[0, 2] / np.sqrt(C[0, 0] * C[2, 2]))))
    print(p, "triples", sum(1 for r in rows if r["panel"] == p) // 2, flush=True)
D = pd.DataFrame(rows); D.to_csv(r"C:/Users/user/tfb/work/g243_flow_noise.csv", index=False)
sq = lambda x: np.sqrt(x.clip(lower=0)); neg = lambda x: int((x < 0).sum())
for lab, d0 in (("strict", D[D.strict]), ("noramp", D)):
    print(f"=== {lab}: sigma in veh/h/lane, median over triples (negative variances counted, taken as 0)")
    for per in ("day", "night", "all"):
        d = d0 if per == "all" else d0[d0.per == per]
        if len(d):
            print(f"  {per:5s} triples {len(d):4d} | three-indicator middle {np.median(sq(d.s2j)):.1f} (neg {neg(d.s2j)}) | first-link form {np.median(sq(d.s2i)):.1f} (neg {neg(d.s2i)}) | adjacent difference {np.median(sq(d.adj)):.1f} | residual sd {d.sd_r.median():.1f} | corr(i, k) {d.corr_ik.median():.3f} | flow {d.q.median():.0f}")
    t = d0.groupby(["panel", "per"]).apply(lambda d: pd.Series(dict(n=len(d), tri=np.median(sq(d.s2j)), adj=np.median(sq(d.adj))))).unstack("per")
    print(t.round(1).to_string())
