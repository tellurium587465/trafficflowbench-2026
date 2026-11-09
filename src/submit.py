"""Assemble the single upload file from three per-task frames, fail-closed.

state  frame: panel,timestamp,link_id,mask_regime,speed_kmh,flow_vph   (station == link here)
queue  frame: window_id,timestamp,link_id,queue_pred
odme   frame: panel,departure_time,path_id,path_flow
"""
from __future__ import annotations
import hashlib, zipfile
from pathlib import Path
import numpy as np, pandas as pd
from common import REL, WORK

KEYPQ = Path(WORK) / "submission_key.parquet"
OUTDIR = Path(WORK) / "subs"
OUTDIR.mkdir(parents=True, exist_ok=True)
EXPECT = {"state": 6_735_795, "queue": 174_000, "odme": 70_708}


def key():
    if not KEYPQ.exists():
        k = pd.read_csv(REL / "submission_key.csv", dtype=str)
        k["submission_id"] = k.submission_id.astype(np.int64)
        k.to_parquet(KEYPQ, index=False)
    return pd.read_parquet(KEYPQ)


def assemble(state: pd.DataFrame, queue: pd.DataFrame, odme: pd.DataFrame, name: str) -> Path:
    k = key()
    out = pd.DataFrame({"submission_id": k.submission_id.to_numpy(), "task": k.task.to_numpy()})
    for c in ("speed_kmh", "flow_vph", "queue_pred", "path_flow"):
        out[c] = 0.0
    specs = {"state": (["panel", "timestamp", "link_id", "mask_regime"], ["speed_kmh", "flow_vph"], state),
             "queue": (["window_id", "timestamp", "link_id"], ["queue_pred"], queue),
             "odme": (["panel", "departure_time", "path_id"], ["path_flow"], odme)}
    for task, (keys, vals, fr) in specs.items():
        m = (k.task == task).to_numpy()
        assert m.sum() == EXPECT[task], (task, m.sum())
        fr = fr.copy()
        for c in keys:
            fr[c] = fr[c].astype(str)
        if "timestamp" in keys and not fr.timestamp.str.match(r"^\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ$").all():
            fr["timestamp"] = pd.to_datetime(fr.timestamp, utc=True).dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        assert not fr.duplicated(keys).any(), f"duplicate keys in {task}"
        j = k.loc[m, keys].merge(fr[keys + vals], on=keys, how="left")
        assert len(j) == m.sum()
        v = j[vals].to_numpy(dtype=float)
        assert np.isfinite(v).all(), f"{task}: {int((~np.isfinite(v)).any(axis=1).sum())} rows missing/non-finite"
        out.loc[m, vals] = v
    st = out.task == "state"
    assert (out.loc[st, "speed_kmh"] > 0).all() and (out.loc[st, "flow_vph"] >= 0).all()
    assert out.loc[out.task == "queue", "queue_pred"].isin([0, 1]).all()
    assert (out.loc[out.task == "odme", "path_flow"] >= 0).all()
    out["queue_pred"] = out.queue_pred.astype(int)
    csv = OUTDIR / f"{name}.csv"
    out.to_csv(csv, index=False, float_format="%.6g", lineterminator="\r\n")   # 10/24: CRLF on every OS (the submitted files were written on Windows)
    z = OUTDIR / f"{name}.zip"
    with zipfile.ZipFile(z, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(csv, arcname="submission.csv")
    csv.unlink()
    print(f"wrote {z} ({z.stat().st_size/1e6:.1f} MB)")
    return z
