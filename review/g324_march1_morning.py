"""G324 (11/05, ledger #257b, measurement only): a first-morning reading of March's map. Per link, the 90 % point of the flow over the visible rows
of March 1, 00:00-09:00 (validation split, masked data) against the same point over the same time of day in the training months, minus 1;
Spearman with G322's head-rate log ratio (c), overall and per panel. Line: overall rho >= 0.2 and the same sign in most panels.
usage: py -B -X utf8 g324_march1_morning.py"""
import sys, numpy as np, pandas as pd
sys.path.insert(0, r"C:/Users/user/tfb_a8/src")
from cachelib import load, T2_PANELS
from scipy.stats import spearmanr
L = pd.read_csv("C:/Users/user/tfb/work/t2/g322_links.csv"); rows = []
for p in T2_PANELS:
    Pt = load(p, "train"); tod = np.arange(Pt.T) % 288; rt = tod < 9 * 12; qt = np.where(np.isfinite(Pt.V[rt]), Pt.Q[rt], np.nan); load.cache_clear()
    Pm = load(p, "validation"); rm = np.arange(9 * 12); qm = np.where(np.isfinite(Pm.V[rm]), Pm.Q[rm], np.nan); load.cache_clear()
    for j in range(Pt.n):
        a, b = qt[:, j], qm[:, j]; a, b = a[np.isfinite(a)], b[np.isfinite(b)]
        rows.append(dict(panel=p, link=j, f=(np.quantile(b, 0.9) / np.quantile(a, 0.9) - 1) if len(a) and len(b) >= 5 else np.nan, nvis=len(b)))
F = pd.DataFrame(rows).merge(L[["panel", "link", "c"]], on=["panel", "link"]).replace([np.inf, -np.inf], np.nan).dropna(subset=["f", "c"])
out = []
for p, d in F.groupby("panel"):
    r = spearmanr(d.c, d.f)[0]; out.append(r); print(f"{p}: links {len(d)}, rho(c, March-1 morning flow p90 ratio) {r:+.3f}, visible rows median {d.nvis.median():.0f}", flush=True)
r = spearmanr(F.c, F.f)[0]; same = sum(np.sign(x) == np.sign(r) for x in out)
print(f"overall rho {r:+.3f} over {len(F)} links, panels with the same sign {same}/{len(out)} | line (rho >= 0.2 and most panels): {r >= 0.2 and same > len(out) / 2}", flush=True)
