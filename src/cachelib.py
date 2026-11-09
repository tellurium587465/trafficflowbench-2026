"""Load the matrix caches and the per-link network parameters the evaluators use."""
from __future__ import annotations
from functools import lru_cache
from pathlib import Path
import os
import numpy as np, pandas as pd
from common import REL, PANELS, WORK

CACHE = Path(os.environ.get("TFB_CACHE", WORK + "/cache"))   # 10/22 E6: per-window entry builds its own
FAMILY = {p: p.rsplit("_", 1)[0] for p in PANELS}
T2_PANELS = [p for p in PANELS if not p.startswith("D12_I405")]


class Panel:
    """One panel/split worth of matrices plus network parameters in link order."""

    def __init__(self, panel: str, split: str):
        self.panel, self.split = panel, split
        z = np.load(CACHE / f"{panel}_{split}.npz", allow_pickle=True)
        if os.environ.get("TFB_EAGER", "0") == "1":
            for k in z.files:
                setattr(self, k, z[k])
        else:                                                     # 10/14 audit D: read an array only when first used
            self._z, self._lazy = z, set(z.files)                 # (same values; 39-73% fewer bytes from the HDD)
            self.times, self.links = z["times"], z["links"]
        self.links = self.links.astype(str)
        self.n = len(self.links)
        self.T = len(self.times)
        nd = REL / "corridors" / panel / "network"
        links = pd.read_csv(nd / "links.csv", dtype={"link_id": str}).drop_duplicates("link_id").set_index("link_id").reindex(self.links)
        fd = pd.read_csv(nd / "fd_parameters.csv", dtype={"link_id": str}).groupby("link_id").mean(numeric_only=True).reindex(self.links)
        self.length = links.length_km.to_numpy(float)          # used by Task 3 (links.csv)
        self.lanes = fd.lanes.to_numpy(float)                    # scoring lanes
        self.vf = fd.free_speed_kmh.to_numpy(float)
        self.cap = fd.capacity_vph.to_numpy(float)
        self.kc = self.cap / np.maximum(self.vf, 1)
        self.kj = fd.k_jam.fillna(100.0 * fd.lanes).to_numpy(float)
        self.kj = np.maximum(self.kj, self.kc * 1.05)
        self.w = self.cap / (self.kj - self.kc)
        # Task 2 threshold: evaluator falls back to 0.6 * links.csv free speed
        self.vcut = 0.6 * pd.to_numeric(links.free_speed_kmh, errors="coerce").fillna(105.0).to_numpy(float)
        topo = pd.read_csv(nd / "lwr_mainline_topology.csv", dtype=str).drop_duplicates("link_id").set_index("link_id").reindex(self.links)
        self.topo = topo
        self.on_attached = (topo.on_ramp_link_ids.fillna("").str.len() > 0).to_numpy()
        self.off_attached = (topo.off_ramp_link_ids.fillna("").str.len() > 0).to_numpy()
        self.day = np.array([t[:10] for t in self.times[::288]])

    def __getattr__(self, name):
        """lazy npz arrays: called only when the attribute is not set yet"""
        if name.startswith("_") or name not in self.__dict__.get("_lazy", ()):
            raise AttributeError(name)
        v = self._z[name]; setattr(self, name, v); return v

    def ramp_validity(self):
        """(T, n) on/off validity flags exactly as score_task3.load_ramp_flows + derive_physics build them."""
        li = {l: i for i, l in enumerate(self.links)}
        on_valid = np.zeros((self.T, self.n), bool); off_valid = np.zeros((self.T, self.n), bool)
        on_flow = np.zeros((self.T, self.n)); off_flow = np.zeros((self.T, self.n))
        for j, (rt, rl) in enumerate(zip(self.ramp_type, self.ramp_link)):
            c = li.get(str(rl))
            if c is None:
                continue
            ok = self.RE[:, j]
            f = np.where(ok, np.nan_to_num(self.R[:, j]), 0.0)
            if str(rt).upper() in ("OR", "ON"):
                on_valid[:, c] |= ok; on_flow[:, c] += f
            elif str(rt).upper() in ("FR", "OFF"):
                off_valid[:, c] |= ok; off_flow[:, c] += f
        on_valid |= ~self.on_attached[None, :]
        off_valid |= ~self.off_attached[None, :]
        return on_valid, off_valid, on_flow, off_flow


@lru_cache(maxsize=None)
def load(panel: str, split: str) -> Panel:
    return Panel(panel, split)


def asof(P, t):
    """10/22 (E6, discussion 749601): a copy of panel P that holds no observation at or after row t - every time-indexed
    float array -> NaN and bool array -> False from row t on (integer masks / regimes are not observations). Task 2 code
    reads the month through this per window, so nothing at or after the forecast origin T can reach a prediction."""
    import copy as _copy
    Pc = _copy.copy(P)
    for k in sorted(P._lazy):
        a = getattr(P, k)
        if isinstance(a, np.ndarray) and a.ndim >= 1 and a.shape[0] == P.T and k != "times":
            if a.dtype.kind == "f":
                b = a.copy(); b[t:] = np.nan; setattr(Pc, k, b)
            elif a.dtype.kind == "b":
                b = a.copy(); b[t:] = False; setattr(Pc, k, b)
    return Pc
