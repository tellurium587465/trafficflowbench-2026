"""G13 = N4 (opposite-direction corridor as a same-day / same-month cue). Train split, the 4 T2 pairs.
A: same day coupling: congested-cell fraction per (date, hour), minus its (weekday/weekend, hour) mean; correlation of
   the residuals between the two directions (incidents or demand redraw shared across directions would make it > 0).
B: month-map coupling: per month, queue frequency by position decile (opposite direction reversed); the month's
   deviation from the 9-month mean; correlation between the two directions' deviation maps."""
import sys, numpy as np, pandas as pd
sys.path.insert(0, r"C:/Users/user/tfb/src")
from cachelib import load
PAIRS = [("D7_I10_E", "D7_I10_W"), ("D7_I210_E", "D7_I210_W"), ("D7_I405_N", "D7_I405_S"), ("D12_I5_N", "D12_I5_S")]


def prep(p, rev):
    P = load(p, "train"); q = (P.V <= P.vcut[None, :]) & np.isfinite(P.V); ok = np.isfinite(P.V)
    t = pd.to_datetime(pd.Series(P.times)); hr = t.dt.floor("h").to_numpy()
    fr = pd.DataFrame({"h": hr, "q": q.sum(1), "n": ok.sum(1)}).groupby("h").sum(); fr = fr.q / fr.n.clip(lower=1)
    x = np.cumsum(np.nan_to_num(P.length)) - np.nan_to_num(P.length) / 2; x = x / x.max()
    if rev:
        x = 1 - x
    dec = np.minimum((x * 10).astype(int), 9); mon = t.dt.strftime("%Y-%m").to_numpy()
    M = {}
    for m in np.unique(mon):
        qm, om = q[mon == m], ok[mon == m]
        f = qm.sum(0) / np.maximum(om.sum(0), 1)
        M[m] = np.array([f[dec == d].mean() if (dec == d).any() else np.nan for d in range(10)])
    return fr, pd.DataFrame(M).T


for a, b in PAIRS:
    fa, Ma = prep(a, False); fb, Mb = prep(b, True)
    j = fa.index.intersection(fb.index); fa, fb = fa[j], fb[j]
    ix = pd.DatetimeIndex(j); key = [(ix.dayofweek >= 5).astype(int), ix.hour]
    ra = fa - fa.groupby(key).transform("mean"); rb = fb - fb.groupby(key).transform("mean")
    pk = (ix.hour >= 6) & (ix.hour <= 20)
    da, db = (Ma - Ma.mean()).to_numpy().ravel(), (Mb - Mb.mean()).to_numpy().ravel(); m = np.isfinite(da) & np.isfinite(db)
    # null for B: shuffle months of the opposite direction
    rng = np.random.default_rng(0); nul = []
    for _ in range(200):
        Ms = (Mb - Mb.mean()).to_numpy()[rng.permutation(len(Mb))].ravel(); mm = np.isfinite(da) & np.isfinite(Ms)
        nul.append(np.corrcoef(da[mm], Ms[mm])[0, 1])
    print(f"{a} vs {b}: A same-hour residual corr {np.corrcoef(ra, rb)[0, 1]:+.3f} (daytime {np.corrcoef(ra[pk], rb[pk])[0, 1]:+.3f}, n={len(j)}) | "
          f"B month-map deviation corr {np.corrcoef(da[m], db[m])[0, 1]:+.3f} (month-shuffle null sd {np.std(nul):.3f})")
