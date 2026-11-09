"""Mean of the three f3 T1 models for a given inference suffix + density clip (v >= q/k_jam).
usage: python t1_ens.py <suffix e.g. _ad2> <out.parquet>"""
import sys, numpy as np, pandas as pd
from cachelib import load
from common import PANELS, WORK

W = WORK + "/t1/"; suf, out = sys.argv[1], sys.argv[2]
import os
MEM = os.environ.get("TFB_T1_ENS", "f3big,f3all,f3all2").split(",")   # 10/10: TFB_T1_ENS=f3nm_a,f3nm_b,f3nm_c for the no-memory models
a, b, c = [pd.read_parquet(W + f"state_{t}{suf}.parquet") for t in MEM]
e = a.copy(); e["speed_kmh"] = (a.speed_kmh + b.speed_kmh + c.speed_kmh) / 3; e["flow_vph"] = (a.flow_vph + b.flow_vph + c.flow_vph) / 3
for p in PANELS:
    P = load(p, "validation"); li = {l: i for i, l in enumerate(P.links)}; m = e.panel.eq(p).values
    ci = e.loc[m, "link_id"].map(li).values
    e.loc[m, "speed_kmh"] = np.maximum(e.loc[m, "speed_kmh"].values, e.loc[m, "flow_vph"].values / P.kj[ci])
e.to_parquet(out, index=False); print("ensemble", len(e), "->", out)
