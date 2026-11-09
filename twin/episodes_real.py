import sys; sys.path.insert(0, r"C:/Users/user/tfb/src")
import numpy as np, pandas as pd
from scipy import ndimage
from cachelib import load
for p in ("D7_I10_E", "D12_I5_S"):
    P = load(p, "train"); vr = P.V / P.vf[None, :]; ok = np.isfinite(vr)
    cong = (P.V <= P.vcut[None, :]) & ok
    # fill single missing cells inside congestion (NaN gaps) for connectivity only
    lab, n = ndimage.label(cong, structure=np.ones((3, 3)))
    sl = ndimage.find_objects(lab); rows = []
    for i, s_ in enumerate(sl, 1):
        m = lab[s_] == i; cells = int(m.sum())
        if cells < 6:
            continue
        sub = vr[s_]; vals = sub[m]
        t0, t1 = s_[0].start, s_[0].stop; l0, l1 = s_[1].start, s_[1].stop
        ext = m.sum(1)                                            # links congested per row
        first = ext[0]; growth = np.diff(ext[:min(len(ext), 7)]) if len(ext) > 1 else np.array([0])
        lv = np.round(np.median(vals), 2); spread = np.quantile(vals, 0.9) - np.quantile(vals, 0.1)
        rows.append(dict(start=t0 % 288, dur=t1 - t0, span=l1 - l0, first=first, g_mean=growth.mean(), lev=lv, spread=spread, head=l1 - 1, traj=ext[:13].tolist()))
    E = pd.DataFrame(rows)
    print(f"== {p}: episodes {len(E)} (>= 6 cells)")
    print("  duration (rows) quantiles 10/50/90:", E.dur.quantile([.1, .5, .9]).tolist(), "| span links:", E.span.quantile([.1, .5, .9]).tolist())
    print("  first-row extent 10/50/90:", E["first"].quantile([.1, .5, .9]).tolist(), "| growth links/row in first 30 min median:", round(E.g_mean.median(), 2))
    print("  within-episode speed spread (q90-q10 of v/vf) median:", round(E.spread.median(), 3), "| level counts:", E.lev.value_counts().head(8).to_dict())
    print("  level by head link (top heads):", E.groupby("head")["lev"].agg(lambda x: x.value_counts().index[0]).head(8).to_dict())
    L = E[E.dur >= 13]; T = np.array([t[:13] for t in L.traj]); print("  extent by row 0..12 (median over episodes lasting >= 13 rows, n=%d):" % len(L), np.median(T, axis=0).tolist())
    load.cache_clear()
