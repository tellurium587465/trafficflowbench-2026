"""FD blend for congested Task 1 cells: v <- (1-a) v + a * q / k_FD, k_FD = k_jam - q / w (congested branch),
for cells predicted below 0.6 v_f (a = 0.75 was the best local strength on 10/6). Density clip kept.

usage: python t1_fd.py <state_in.parquet> <state_out.parquet> [a]
"""
import sys, numpy as np, pandas as pd
from cachelib import load, PANELS

a = float(sys.argv[3]) if len(sys.argv) > 3 else 0.75
st = pd.read_parquet(sys.argv[1]); n = 0
for p in PANELS:
    P = load(p, "validation"); li = {l: i for i, l in enumerate(P.links)}
    m = st.panel.eq(p).values; c = st.loc[m, "link_id"].map(li).values
    v = st.loc[m, "speed_kmh"].values; q = st.loc[m, "flow_vph"].values
    cong = v < 0.6 * P.vf[c]; k = np.maximum(P.kj[c] - q / P.w[c], 1e-3)
    vn = np.where(cong, (1 - a) * v + a * q / k, v); vn = np.maximum(vn, q / P.kj[c])
    n += int(cong.sum()); st.loc[m, "speed_kmh"] = vn
st.to_parquet(sys.argv[2], index=False); print("congested cells blended", n)
