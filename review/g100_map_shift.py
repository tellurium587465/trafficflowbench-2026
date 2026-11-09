"""G100 (10/24, report Fig. 1): how the congestion map moves between the training months and March. Per link, the share of
observed rows that are queued (v <= vcut), training months (unmasked layer) vs March (released masked layer, observed rows
only); per corridor the Spearman rank correlation of the link shares and the total-variation distance of the normalised
maps. Writes work/t2/g100_map_shift.csv (link level) and prints the per-corridor table.
usage: python g100_map_shift.py"""
import sys
sys.path.insert(0, r"C:/Users/user/tfb/src")
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from cachelib import load, T2_PANELS
rows, tab = [], []
for p in T2_PANELS:
    sh = {}
    for sp in ("train", "validation"):
        P = load(p, sp); obs = np.isfinite(P.V); q = (P.V <= P.vcut[None, :]) & obs
        sh[sp] = q.sum(0) / np.maximum(obs.sum(0), 1); load.cache_clear()
    a, b = sh["train"], sh["validation"]; rho = spearmanr(a, b).correlation
    tv = 0.5 * np.abs(a / max(a.sum(), 1e-9) - b / max(b.sum(), 1e-9)).sum()
    tab.append(dict(panel=p, links=len(a), rho=rho, tv=tv, train_mean=a.mean(), march_mean=b.mean()))
    rows += [dict(panel=p, link=i, train=x, march=y) for i, (x, y) in enumerate(zip(a, b))]
pd.DataFrame(rows).to_csv(r"C:/Users/user/tfb/work/t2/g100_map_shift.csv", index=False)
print(pd.DataFrame(tab).round(3).to_string(index=False))
