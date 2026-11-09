"""Shared loaders for TrafficFlowBench work."""
from __future__ import annotations
import glob, json, os
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]                   # 10/23 portability: the repository root (src/..)
WORK = str(Path(os.environ.get("TFB_WORK", ROOT / "work"))).replace("\\", "/").rstrip("/")   # work files (probs, LightGBM, subs)
REL = Path(os.environ.get("TFB_REL", ROOT / "data" / "kaggle_public"))
# NN data / models (10/22: moved off the failing E: HDD to C:/tfb_data, copied from the verified D:/tfb_mirror)
NN = os.environ.get("TFB_NN", r"C:/tfb_data/tfb_nn").replace("\\", "/").rstrip("/")
PANELS = [p["corridor_id"] for p in json.loads((REL/"config"/"corridors.json").read_text())["panels"]]

def network(panel):
    nd = REL/"corridors"/panel/"network"
    links = pd.read_csv(nd/"links.csv", dtype={"link_id": str})
    fd = pd.read_csv(nd/"fd_parameters.csv", dtype={"link_id": str})
    topo = pd.read_csv(nd/"lwr_mainline_topology.csv", dtype=str)
    ramps = pd.read_csv(nd/"ramp_attachment_map.csv", dtype=str)
    return links, fd, topo, ramps

def day_files(panel, split, layer="mainline_states"):
    return sorted(glob.glob(str(REL/"corridors"/panel/split/layer/"**"/"*.parquet"), recursive=True))

def load_days(panel, split, layer="mainline_states", cols=None, files=None):
    fs = files if files is not None else day_files(panel, split, layer)
    return pd.concat([pd.read_parquet(f, columns=cols) for f in fs], ignore_index=True)

def link_order(panel, split="train"):
    f = day_files(panel, split, "mainline_states" if split=="train" else "mainline_states_masked")[0]
    d = pd.read_parquet(f, columns=["link_id","milepost"]).drop_duplicates("link_id")
    d["link_id"] = d.link_id.astype(str)
    return d.sort_values(["milepost","link_id"]).link_id.tolist(), d.set_index("link_id").milepost

def to_matrix(df, links, col, times=None):
    times = times if times is not None else sorted(df.timestamp.unique())
    m = df.pivot_table(index="timestamp", columns="link_id", values=col, aggfunc="mean").reindex(index=times, columns=links)
    return m.to_numpy(dtype=float), times
