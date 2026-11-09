"""Build per-panel, per-split matrix caches (time x link) from the release.

One file per (panel, split): work/cache/<panel>_<split>.npz holding
  times      (T,) str, contiguous 5-minute grid over the split
  links      (n,) str, milepost order
  V, Q, O    float32 speed / flow / occupancy as released (NaN where absent)
             train: the unmasked layer; validation/private: the masked layer
  P          uint8 pct_observed
  E          bool  is_score_eligible
  regime     (T,) uint8 0/1/2 for R1/R2/R3 (day's mask regime), 255 if unknown
  target     bool  Task 1 target cells, read from the official template
  blank      bool  masked-layer blanks (train only: target + Task 2 horizon/buffer)
  R          float32 (T, n_ramps) ramp flows; RE bool ramp eligible
  ramp_ids, ramp_type, ramp_link
"""
from __future__ import annotations
import sys, glob, os
from pathlib import Path
import numpy as np, pandas as pd
from common import REL, PANELS, link_order, WORK

OUT = Path(os.environ.get("TFB_CACHE", WORK + "/cache"))     # 10/22 E6: same switch as cachelib.CACHE
OUT.mkdir(parents=True, exist_ok=True)
SPLITS = {"train": ("2030-06-01", "2031-03-01"), "validation": ("2031-03-01", "2031-04-01"), "private": ("2031-04-01", "2031-05-01")}
for _s in filter(None, os.environ.get("TFB_SPLIT_DATES", "").split(";")):   # 11/05 ledger #246, a newly generated month: e.g. "private=2031-05-01,2031-06-01"
    _k, _v = _s.split("="); _a, _b = _v.split(","); SPLITS[_k.strip()] = (_a.strip(), _b.strip())                 # (unset = the dates above, unchanged)
REG = {"R1": 0, "R2": 1, "R3": 2}


def grid(split):
    a, b = SPLITS[split]
    t = pd.date_range(a, b, freq="5min", inclusive="left", tz="UTC")
    return t.strftime("%Y-%m-%dT%H:%M:%SZ").to_numpy()


def build(panel, split):
    order, _ = link_order(panel, "train")
    li = {l: i for i, l in enumerate(order)}
    times = grid(split); ti = {t: i for i, t in enumerate(times)}
    T, n = len(times), len(order)
    V = np.full((T, n), np.nan, np.float32); Q = V.copy(); O = V.copy()
    P = np.zeros((T, n), np.uint8); E = np.zeros((T, n), bool)
    blank = np.zeros((T, n), bool); regime = np.full(T, 255, np.uint8)
    root = REL / "corridors" / panel / split
    cols = ["timestamp", "link_id", "speed_kmh", "flow_vph", "occupancy", "pct_observed", "is_score_eligible"]
    # released observation layer
    if split == "train":
        files = sorted(glob.glob(str(root / "mainline_states" / "**" / "*.parquet"), recursive=True))
        for f in files:
            d = pd.read_parquet(f, columns=cols)
            r = d.timestamp.map(ti).to_numpy(); c = d.link_id.astype(str).map(li).to_numpy()
            V[r, c] = d.speed_kmh; Q[r, c] = d.flow_vph; O[r, c] = d.occupancy
            P[r, c] = d.pct_observed; E[r, c] = d.is_score_eligible.astype(bool)
    for reg, code in REG.items():
        for f in sorted(glob.glob(str(root / "mainline_states_masked" / f"mask_regime={reg}" / "*.parquet"))):
            d = pd.read_parquet(f, columns=cols)
            r = d.timestamp.map(ti).to_numpy(); c = d.link_id.astype(str).map(li).to_numpy()
            regime[np.unique(r)] = code
            if split == "train":
                blank[r, c] = d.speed_kmh.isna().to_numpy() & np.isfinite(V[r, c])
            else:
                V[r, c] = d.speed_kmh; Q[r, c] = d.flow_vph; O[r, c] = d.occupancy
                P[r, c] = d.pct_observed; E[r, c] = d.is_score_eligible.astype(bool)
    # official Task 1 targets
    target = np.zeros((T, n), bool)
    tp = REL / "task1" / panel / split / "sample_submission_state.csv"
    t = pd.read_csv(tp, usecols=["timestamp", "link_id", "mask_regime"], dtype=str)
    r = t.timestamp.map(ti).to_numpy(); c = t.link_id.map(li).to_numpy()
    assert not np.isnan(r.astype(float)).any() and not np.isnan(c.astype(float)).any()
    target[r.astype(int), c.astype(int)] = True
    # ramps
    amap = pd.read_csv(REL / "corridors" / panel / "network" / "ramp_attachment_map.csv", dtype=str)
    rids = amap.ramp_link_id.tolist(); ri = {x: i for i, x in enumerate(rids)}
    R = np.full((T, len(rids)), np.nan, np.float32); RE = np.zeros((T, len(rids)), bool)
    for f in sorted(glob.glob(str(root / "ramp_states" / "**" / "*.parquet"), recursive=True)):
        d = pd.read_parquet(f, columns=["timestamp", "ramp_link_id", "flow_vph", "pct_observed", "is_score_eligible"])
        r = d.timestamp.map(ti).to_numpy(); c = d.ramp_link_id.astype(str).map(ri)
        ok = c.notna().to_numpy(); c = c[ok].astype(int).to_numpy(); r = r[ok]
        R[r, c] = d.flow_vph.to_numpy()[ok]
        RE[r, c] = (d.is_score_eligible.astype(bool) & (d.pct_observed >= 75) & d.flow_vph.notna()).to_numpy()[ok]
    np.savez(OUT / f"{panel}_{split}.npz", times=times, links=np.array(order), V=V, Q=Q, O=O, P=P, E=E,
             regime=regime, target=target, blank=blank, R=R, RE=RE, ramp_ids=np.array(rids),
             ramp_type=amap.ramp_type.to_numpy(), ramp_link=amap.nearest_mainline_link_id.to_numpy())
    print(panel, split, "T", T, "n", n, "targets", int(target.sum()), "regimes", np.bincount(regime[regime < 3], minlength=3) // 288, flush=True)


if __name__ == "__main__":
    panels = sys.argv[1:] or PANELS
    for p in panels:
        for s in SPLITS:
            build(p, s)
