"""Ensemble the split-adaptive T1 predictions (mean of f3big/f3all/f3all2), clip density at jam, optional FD blend,
and assemble submissions that differ from s17 only in Task 1."""
import numpy as np, pandas as pd
from cachelib import load
from common import PANELS
from submit import assemble

W = "C:/Users/user/tfb/work/"
a, b, c = [pd.read_parquet(W + f"t1/state_{t}_ad.parquet") for t in ("f3big", "f3all", "f3all2")]
e = a.copy(); e["speed_kmh"] = (a.speed_kmh + b.speed_kmh + c.speed_kmh) / 3; e["flow_vph"] = (a.flow_vph + b.flow_vph + c.flow_vph) / 3
parts, partsf = [], []
for p in PANELS:
    P = load(p, "validation"); li = {l: i for i, l in enumerate(P.links)}
    x = e[e.panel == p].copy(); ci = x.link_id.map(li).values
    v = x.speed_kmh.values; q = x.flow_vph.values
    v = np.maximum(v, q / P.kj[ci])                                # jam clip: k = q/v <= k_jam
    x["speed_kmh"] = v; parts.append(x)
    y = x.copy(); cong = v < 0.6 * P.vf[ci]; k = np.maximum(P.kj[ci] - q / P.w[ci], 1e-3)
    y["speed_kmh"] = np.where(cong, 0.25 * v + 0.75 * q / k, v); partsf.append(y)
ens = pd.concat(parts).loc[e.index]; ensf = pd.concat(partsf).loc[e.index]
ens.to_parquet(W + "t1/state_f3ens3c_ad.parquet", index=False); ensf.to_parquet(W + "t1/state_f3ens3c_fdv075_ad.parquet", index=False)
old = pd.read_parquet(W + "t1/state_f3ens3c.parquet")
print("mean |dv| vs old %.3f km/h, mean |dq| %.2f vph" % ((ens.speed_kmh.values - old.speed_kmh.values).__abs__().mean(), (ens.flow_vph.values - old.flow_vph.values).__abs__().mean()))
q = pd.read_parquet(W + "t2/queue_s17_t1f3ens3c_t2v12nppdem_nnDnp6_t4l2.parquet")
od = pd.read_parquet(W + "t4/odme_l2.parquet")
assemble(ens, q, od, "s18_t1ens3cAdapt_t2s17_t4l2")
assemble(ensf, q, od, "s18f_t1ens3cAdaptFd075_t2s17_t4l2")
