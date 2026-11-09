"""G11 by day type (P5): the fixed rule (A=1.0, B=0.85, dil 1) vs gate only, official-like ongoing windows,
weekday / weekend and month-start days (1-4) separately."""
import sys, numpy as np, pandas as pd
sys.argv = [sys.argv[0]]
exec(open("g11_ongoing_evidence.py", encoding="utf-8").read().split("RULES = [None]")[0])
dt = np.array([pd.Timestamp(str(T.META["date"][w["k"]])[:10]) for w in W])
wk = np.array([d.dayofweek < 5 for d in dt]); ms = np.array([d.day <= 4 for d in dt]); off = np.array([w["off"] for w in W])
cl = np.array([w["cl"] for w in W])
d = np.array([score(w, (1.0, 0.85, 1)) - score(w, None) for w in W])
for name, m in (("all", off), ("weekday", off & wk), ("weekend", off & ~wk), ("days 1-4", off & ms), ("weekday days 1-4", off & wk & ms)):
    idx = np.flatnonzero(m); u = np.unique(cl[idx]); g = {c: idx[cl[idx] == c] for c in u}
    se = np.std([d[np.concatenate([g[c] for c in rng.choice(u, len(u))])].mean() for _ in range(300)])
    print(f"{name:17s} n={len(idx):4d}  {d[idx].mean():+.4f} (SE {se:.4f})")
