"""What separates the generator's blocks? (10/14, follows block_structure.py) For every adjacent link pair seen congested
>= 30 rows: tight (split rate <= 0.05, one shared state) or not, against the released network: lane change, FD parameter
change (fd_parameters: free speed, capacity, k_jam), connectors / ramps at the junction (lwr_mainline_topology), and
the two link lengths. If one network attribute separates tight from split pairs, the blocks can be read from the
network (no estimation, so they hold in the test months too).
usage: python block_boundaries.py"""
import sys, numpy as np, pandas as pd
sys.path.insert(0, r"C:/Users/user/tfb/src")
from cachelib import load
from common import REL

B = pd.read_csv(r"C:/Users/user/tfb/work/twin_blocks.csv"); rows = []
for p, g in B.groupby("panel"):
    P = load(p, "train"); links = [str(x) for x in P.links]
    net = REL / "corridors" / p / "network"
    fd = pd.read_csv(net / "fd_parameters.csv", dtype={"link_id": str}).groupby("link_id").mean(numeric_only=True)
    tp = pd.read_csv(net / "lwr_mainline_topology.csv", dtype=str).drop_duplicates("link_id").set_index("link_id")
    for r in g.itertuples():
        if r.n_cong < 30 or not np.isfinite(r.split):
            continue
        a, b = links[r.i], links[r.i + 1]
        fa, fb = fd.reindex([a, b]).to_numpy(float)
        ta, tb = tp.reindex([a]).iloc[0], tp.reindex([b]).iloc[0]
        out_a = [x for x in str(ta.get("outgoing_link_ids", "")).split(";") if x and x != "nan"]
        in_b = [x for x in str(tb.get("incoming_link_ids", "")).split(";") if x and x != "nan"]
        rows.append(dict(panel=p, i=r.i, tight=r.split <= 0.05, split=r.split,
                         lanes_chg=fa[list(fd.columns).index("lanes")] != fb[list(fd.columns).index("lanes")],
                         vf_chg=abs(fa[list(fd.columns).index("free_speed_kmh")] - fb[list(fd.columns).index("free_speed_kmh")]) > 0.05,
                         cap_chg=abs(fa[list(fd.columns).index("capacity_vph")] - fb[list(fd.columns).index("capacity_vph")]) > 1.0,
                         kj_chg=abs(fa[list(fd.columns).index("k_jam")] - fb[list(fd.columns).index("k_jam")]) > 0.5,
                         conn_out_a=any(x.startswith("CONN") for x in out_a), conn_in_b=any(x.startswith("CONN") for x in in_b),
                         offramp_a=str(ta.get("off_ramp_link_ids", "nan")) not in ("", "nan"),
                         onramp_b=str(tb.get("on_ramp_link_ids", "nan")) not in ("", "nan"),
                         sens_a=str(ta.get("has_sensor", "")), sens_b=str(tb.get("has_sensor", "")),
                         len_a=r.len_i, len_b=float(P.length[r.i + 1])))
    load.cache_clear()
R = pd.DataFrame(rows); R.to_csv(r"C:/Users/user/tfb/work/twin_block_boundaries.csv", index=False)
print("pairs", len(R), "tight share", round(R.tight.mean(), 3))
for c in ["lanes_chg", "vf_chg", "cap_chg", "kj_chg", "conn_out_a", "conn_in_b", "offramp_a", "onramp_b"]:
    t = R.groupby(c).tight.agg(["mean", "size"]).round(3)
    print(f"{c:11s}", {bool(k): (float(v["mean"]), int(v["size"])) for k, v in t.iterrows()})
R["any_param_chg"] = R.lanes_chg | R.vf_chg | R.cap_chg | R.kj_chg
print("any FD/lane change", R.groupby("any_param_chg").tight.agg(["mean", "size"]).round(3).to_dict("index"))
print("by panel: tight share | P(tight | no param change) | P(tight | param change)")
for p, g in R.groupby("panel"):
    print(f"  {p}: {g.tight.mean():.2f} | {g[~g.any_param_chg].tight.mean():.2f} ({(~g.any_param_chg).sum()}) | {g[g.any_param_chg].tight.mean():.2f} ({g.any_param_chg.sum()})")
