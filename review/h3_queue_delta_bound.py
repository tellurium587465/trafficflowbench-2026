"""Harness review H3: before spending an LB slot, how much of a T2 change is even visible on the public split?
usage: python h3_queue_delta_bound.py <a.zip> <b.zip>
Per split: windows touched, changed cells, and a crude per-window bound |dIoU| <= changed / max(|a|,|b|,1)
turned into a bound on |d total| = 0.30 * mean_w(bound) (S_queue is a mean over windows; crude, ignores
panel/family weighting). Read-only."""
import sys, time, zipfile, numpy as np, pandas as pd
t0 = time.time()
key = pd.read_parquet(r"C:/Users/user/tfb/work/submission_key.parquet", columns=["task", "split", "window_id"])
m = (key.task == "queue").to_numpy()
def rd(z):
    with zipfile.ZipFile(z) as zf:
        return pd.read_csv(zf.open("submission.csv"), usecols=["task", "queue_pred"], dtype={"task": "category", "queue_pred": np.int8}).queue_pred.to_numpy()[m]
qa, qb = rd(sys.argv[1]), rd(sys.argv[2])
d = pd.DataFrame({"split": key.split.to_numpy()[m], "w": key.window_id.to_numpy()[m], "a": qa, "b": qb, "ch": (qa != qb).astype(int)})
g = d.groupby(["split", "w"]).agg(a=("a", "sum"), b=("b", "sum"), ch=("ch", "sum")).reset_index()
g["bound"] = g.ch / np.maximum(np.maximum(g.a, g.b), 1)
for s, x in g.groupby("split"):
    print(f"{s:10s} windows {len(x)}  touched {int((x.ch > 0).sum())}  changed cells {int(x.ch.sum())}  "
          f"mean per-window bound {x.bound.mean():.4f}  => |d total| <= {0.30 * x.bound.mean():.4f}")
print(f"{time.time()-t0:.1f}s")
