"""Harness review H2: minimal submission diff / sanity report (prototype of a pre-submit gate).
usage: python h2_sub_diff.py <a.zip> <b.zip>
Prints, per task, how many rows changed and by how much, plus value-range sanity for b.
Rows are aligned by position (submit.assemble writes them in submission_key order). Read-only."""
import sys, time, zipfile, numpy as np, pandas as pd
t0 = time.time()
COLS = {"submission_id": np.int64, "task": "category", "speed_kmh": np.float32, "flow_vph": np.float32,
        "queue_pred": np.int8, "path_flow": np.float64}
def rd(z):
    with zipfile.ZipFile(z) as zf:
        return pd.read_csv(zf.open("submission.csv"), dtype=COLS)
a, b = rd(sys.argv[1]), rd(sys.argv[2])
assert len(a) == len(b) and (a.submission_id.values == b.submission_id.values).all(), "row order differs"
for task, cols in (("state", ["speed_kmh", "flow_vph"]), ("queue", ["queue_pred"]), ("odme", ["path_flow"])):
    m = (b.task == task).values
    for c in cols:
        x, y = a.loc[m, c].to_numpy(np.float64), b.loc[m, c].to_numpy(np.float64)
        d = np.abs(x - y); ch = int((d > 1e-6 * np.maximum(1, np.abs(x))).sum())
        print(f"{task:5s} {c:10s} rows {m.sum():>9,d} changed {ch:>9,d} ({ch/m.sum():.2%})  mean|d| {d.mean():.4g}  max|d| {d.max():.4g}"
              f"  b: min {y.min():.4g} max {y.max():.4g} mean {y.mean():.4g}")
qa, qb = a.loc[a.task == "queue", "queue_pred"].to_numpy(), b.loc[b.task == "queue", "queue_pred"].to_numpy()
print(f"queue positives a {qa.sum():,d}  b {qb.sum():,d}  cell agreement {(qa == qb).mean():.4%}  "
      f"IoU(a,b) on positives {((qa & qb).sum() / max((qa | qb).sum(), 1)):.4f}")
print(f"{time.time()-t0:.1f}s")
