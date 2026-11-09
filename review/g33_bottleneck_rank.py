"""G33: is onset location physics-identifiable (premise for a generator-identification / simulation forecast)?
Train selector-like onset windows (t2_eval_surprise.SEL). True site = label links at T+30; bottleneck b = the link just
downstream of the most downstream label link (higher index = downstream, 10/10 audit). Per link at T:
  u_fd  = mean observed flow over T-6..T-1 / fd capacity
  u_emp = same / empirical capacity (99.5th pct of the 15-min mean flow over all train days, per link)
Reports the rank of b among links by u (1 = highest), and the share of windows where b is within the top 1/3/5, plus
the same for the label links themselves (does the queue sit at the highest-u link instead?).
usage: python g33_bottleneck_rank.py"""
import sys, numpy as np, pandas as pd, warnings
warnings.simplefilter("ignore")
sys.path.insert(0, r"C:/Users/user/tfb/src")
import t2_eval_surprise as S
from t2_eval_oof import T
from cachelib import load
EC = {}


def emp_cap(p):
    if p not in EC:
        P = load(p, "train"); D = P.T // 288
        q = P.Q[:D * 288].reshape(D * 96, 3, P.n); m = np.nanmean(q, axis=1)
        EC[p] = (P, np.nanpercentile(m, 99.5, axis=0))
    return EC[p]


rows = []
for k in S.SEL:
    p = str(T.META["panel"][k]); t = int(T.META["t"][k]); P, ce = emp_cap(p)
    lab, el = T.LAB[k]; y = (lab[-1] & el[-1]).astype(bool)
    if not y.any():
        continue
    qb = np.nanmean(P.Q[t - 6:t], axis=0)
    b = min(int(np.flatnonzero(y).max()) + 1, P.n - 1)
    r = dict(panel=p, n=P.n, f=S.F[S.SEL.index(k)])
    for nm, cap in (("fd", P.cap), ("emp", ce)):
        u = np.nan_to_num(qb / cap, nan=-1)
        order = np.argsort(-u); rank = np.empty(P.n, int); rank[order] = np.arange(1, P.n + 1)
        r[f"rank_b_{nm}"] = rank[b]; r[f"rank_lab_{nm}"] = rank[y].min(); r[f"u_b_{nm}"] = u[b]; r[f"u_max_{nm}"] = u.max()
    rows.append(r)
R = pd.DataFrame(rows)
print(f"windows {len(R)}, links per panel median {R.n.median():.0f}")
for nm in ("fd", "emp"):
    for w in ("b", "lab"):
        x = R[f"rank_{w}_{nm}"]
        print(f"cap={nm:3s} {w:3s}: rank median {x.median():.0f}, top1 {np.mean(x <= 1):.3f}, top3 {np.mean(x <= 3):.3f}, top5 {np.mean(x <= 5):.3f}, top10 {np.mean(x <= 10):.3f}")
    print(f"   u at b median {R[f'u_b_{nm}'].median():.3f} vs max u median {R[f'u_max_{nm}'].median():.3f}")
R["rare"] = R.f <= 0.3
print(R.groupby("rare")[["rank_b_emp", "rank_lab_emp"]].median())
