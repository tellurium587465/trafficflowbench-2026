"""Audit A11: link order used everywhere (milepost sort of the first train file) vs the released topology
(lwr_mainline_topology order_index and incoming/outgoing links); cumulative x vs milepost; stations per link."""
import sys, glob, numpy as np, pandas as pd
sys.path.insert(0, r"C:/Users/user/tfb/src")
from common import REL, PANELS, link_order

for p in PANELS:
    order, mp = link_order(p)
    topo = pd.read_csv(REL / "corridors" / p / "network" / "lwr_mainline_topology.csv", dtype=str).drop_duplicates("link_id").set_index("link_id")
    oi = pd.to_numeric(topo.order_index.reindex(order), errors="coerce").to_numpy()
    mono = np.all(np.diff(oi[np.isfinite(oi)]) > 0)
    # successor consistency: the next link in our order should be among the outgoing links of the current one
    inset = set(order); bad = 0; chk = 0
    for a, b in zip(order[:-1], order[1:]):
        outs = [x for x in str(topo.loc[a].outgoing_link_ids).split(";") if x in inset] if a in topo.index else []
        if outs:
            chk += 1; bad += b not in outs
    links = pd.read_csv(REL / "corridors" / p / "network" / "links.csv", dtype={"link_id": str}).drop_duplicates("link_id").set_index("link_id").reindex(order)
    x = np.concatenate([[0.0], np.cumsum(links.length_km.to_numpy()[:-1])])
    m = mp.reindex(order).to_numpy(); m = m - m[0]
    f = sorted(glob.glob(str(REL / "corridors" / p / "validation" / "mainline_states_masked" / "**" / "*.parquet"), recursive=True))[0]
    d = pd.read_parquet(f, columns=["station_id", "link_id"])
    print(f"{p}: n={len(order)} order_index monotone={mono} next-link-is-outgoing {chk-bad}/{chk}; cum length end {x[-1]:.1f} km vs milepost span {m[-1]:.1f}; "
          f"max|x - milepost| {np.nanmax(np.abs(x - m)):.2f} km; stations/link max {d.groupby('link_id').station_id.nunique().max()}")
