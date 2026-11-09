"""Twin fidelity for site choice (the lever for sim variety, 10/13): world A replays every real train day with that
day's upstream demand. Per panel-day, the first morning queue (06:00-12:00, truth layer: v <= vcut on eligible cells)
in the real data vs in twin A: same site (head = most downstream queued link, within +-2) and time difference.
Also the dominance: share of days won by each panel's most frequent site, real vs twin (F57: twin onsets less regular).
usage: python fidelity_site.py [world=A]"""
import sys, numpy as np, pandas as pd, warnings
warnings.simplefilter("ignore")
sys.path.insert(0, r"C:/Users/user/tfb/src")
from cachelib import load, T2_PANELS
W = sys.argv[1] if len(sys.argv) > 1 else "A"


def first(P):
    truth = (np.nan_to_num(P.V, nan=1e9) <= P.vcut[None, :]) & P.E
    out = {}
    for d in range(P.T // 288):
        for s in range(72, 144):
            r = d * 288 + s
            if truth[r].any():
                out[str(P.day[d])] = (s, int(np.flatnonzero(truth[r]).max())); break
    return out


rows = []
for p in T2_PANELS:
    a, b = first(load(p, "train")), first(load(p, f"twin{W}"))
    common = sorted(set(a) & set(b))
    same = [abs(a[d][1] - b[d][1]) <= 2 for d in common]; dt = [abs(a[d][0] - b[d][0]) for d in common]
    ha = pd.Series([a[d][1] // 3 for d in common]).value_counts(normalize=True); hb = pd.Series([b[d][1] // 3 for d in common]).value_counts(normalize=True)
    rows.append(dict(panel=p, days=len(common), only_real=len(set(a) - set(b)), only_twin=len(set(b) - set(a)),
                     same_site=np.mean(same), med_dt_min=5 * np.median(dt), dom_real=ha.iloc[0], dom_twin=hb.iloc[0],
                     top_real=ha.index[0] * 3, top_twin=hb.index[0] * 3))
R = pd.DataFrame(rows); print(R.round(3).to_string(index=False))
print(f"\nmean same-site {R.same_site.mean():.3f}; dominance real {R.dom_real.mean():.3f} vs twin {R.dom_twin.mean():.3f}")
