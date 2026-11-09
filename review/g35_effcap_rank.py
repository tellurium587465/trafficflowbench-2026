"""G35: rank of the true bottleneck by utilization against the EFFECTIVE breakdown flow (G34 events, leave-month-out)
instead of the fd capacity. C_eff(link) = median breakdown flow of the morning first-queue events at that link in the
other train months (>= 3 events); links without events: censored, C = 1.05 x the 99.5th pct of the 30-min mean flow
(other months). Flow at link i = max over i..i+2 of the mean over T-6..T-1 (as G34). Onset windows = SEL (G33).
usage: python g35_effcap_rank.py"""
import sys, numpy as np, pandas as pd, warnings
warnings.simplefilter("ignore")
sys.path.insert(0, r"C:/Users/user/tfb/src")
import t2_eval_surprise as S
from t2_eval_oof import T
from cachelib import load
E = pd.read_csv(r"C:/Users/user/tfb/work/t2/g34_events.csv")
PC, CC = {}, {}


def panel(p):
    if p not in PC:
        P = load(p, "train"); D = P.T // 288
        mon = np.array([str(d)[:7] for d in P.day[:D]])
        m30 = np.nanmean(P.Q[:D * 288].reshape(D, 48, 6, P.n), axis=2)                 # (D, 48, n) 30-min means
        PC[p] = (P, mon, m30)
    return PC[p]


def ceff(p, month):
    key = (p, month)
    if key not in CC:
        P, mon, m30 = panel(p)
        hi = np.nanpercentile(m30[mon != month].reshape(-1, P.n), 99.5, axis=0) * 1.05
        e = E[(E.panel == p) & (np.array([mon[d] for d in E[E.panel == p].day]) != month if (E.panel == p).any() else [])] if False else None
        Ep = E[E.panel == p]; Ep = Ep[np.array([mon[d] != month for d in Ep.day])]
        c = hi.copy(); known = np.zeros(P.n, bool)
        for b, g in Ep.groupby("b"):
            if len(g) >= 3:
                c[b] = np.nanmedian(g.bf); known[b] = True
        CC[key] = (c, known)
    return CC[key]


def flow3(P, t):
    f = np.nanmean(P.Q[t - 6:t], axis=0); out = np.full(P.n, np.nan)
    for i in range(P.n):
        out[i] = np.nanmax(f[i:i + 3]) if np.isfinite(f[i:i + 3]).any() else np.nan
    return out


rows = []
for k in S.SEL:
    p = str(T.META["panel"][k]); t = int(T.META["t"][k]); P, mon, _ = panel(p)
    lab, el = T.LAB[k]; y = (lab[-1] & el[-1]).astype(bool)
    if not y.any():
        continue
    b = min(int(np.flatnonzero(y).max()) + 1, P.n - 1)
    c, known = ceff(p, str(T.META["date"][k])[:7])
    u = np.nan_to_num(flow3(P, t) / c, nan=-1)
    order = np.argsort(-u); rank = np.empty(P.n, int); rank[order] = np.arange(1, P.n + 1)
    # rank among known sites only; review 10/13: known links within +-2 are ONE physical bottleneck (89% of the sites
    # had a neighbour site), so merge them into clusters, score a cluster by its max u, and count b's cluster as hit
    # when b lies within +-2 of it
    ks = np.flatnonzero(known); cl = []
    for i in ks:
        if cl and i - cl[-1][-1] <= 2:
            cl[-1].append(i)
        else:
            cl.append([i])
    uc = np.array([u[c_].max() for c_ in cl]); hitc = [j for j, c_ in enumerate(cl) if min(abs(b - x) for x in c_) <= 2]
    rk = int((uc > uc[hitc[0]]).sum() + 1) if hitc else np.nan
    rows.append(dict(rank=rank[b], known=bool(hitc), rank_known=rk, nknown=len(cl), u_b=u[b], u_max=u.max(), f=S.F[S.SEL.index(k)],
                     morning=(t + 6) % 288 < 144))
R = pd.DataFrame(rows)
R = R[R.morning]                                    # review 10/13: the official onset windows are morning windows only
print(f"(morning windows only) windows {len(R)}; bottleneck has an effective capacity from other months: {R.known.mean():.3f}; known sites per panel median {R.nknown.median():.0f}")
x = R["rank"]; print(f"rank among all links: median {x.median():.0f}, top1 {np.mean(x <= 1):.3f}, top3 {np.mean(x <= 3):.3f}, top5 {np.mean(x <= 5):.3f}")
K = R[R.known]; x = K.rank_known
print(f"known bottlenecks: rank among known sites median {x.median():.0f}, top1 {np.mean(x <= 1):.3f}, top2 {np.mean(x <= 2):.3f}, top3 {np.mean(x <= 3):.3f}; u_b median {K.u_b.median():.3f}")
print(R.assign(rare=R.f <= 0.3).groupby("rare")[["known", "rank"]].agg(["mean", "median"]).round(3))
