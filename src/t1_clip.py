"""Clip Task 1 speeds at the per-link observed ceiling (traffic review: speeds are capped, e.g. 101.2 km/h on
11 D7_I10 links with no noise above; our predictions exceed the link maximum in ~1% of rows, up to +26 km/h).
Ceiling = max(train maximum, the split's own released maximum) per link. Density clip v >= q/k_jam kept.

usage: python t1_clip.py <state_in.parquet> <state_out.parquet>
"""
import sys, numpy as np, pandas as pd
from cachelib import load, PANELS

st = pd.read_parquet(sys.argv[1]); n_clip = 0
for p in PANELS:
    T = load(p, "train"); li = {l: i for i, l in enumerate(T.links)}
    cap_v = np.nanmax(np.where(np.isfinite(T.V), T.V, np.nan), axis=0)
    for split in ("validation", "private"):
        S = load(p, split); assert list(S.links) == list(T.links)
        cap_v = np.fmax(cap_v, np.nanmax(np.where(np.isfinite(S.V), S.V, np.nan), axis=0))
    m = st.panel.eq(p).values; c = st.loc[m, "link_id"].map(li).values
    v = st.loc[m, "speed_kmh"].values; q = st.loc[m, "flow_vph"].values
    vc = np.minimum(v, cap_v[c]); vc = np.maximum(vc, q / T.kj[c])       # keep k <= k_jam
    n_clip += int((vc < v).sum()); st.loc[m, "speed_kmh"] = vc
st.to_parquet(sys.argv[2], index=False)
print("clipped rows", n_clip, "of", len(st))
