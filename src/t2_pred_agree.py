"""Per-window agreement (IoU of predicted sets) between two queue parquets, by condition and day of month."""
import sys, numpy as np, pandas as pd
from common import REL
from cachelib import T2_PANELS

W = "C:/Users/user/tfb/work/t2/"
a = pd.read_parquet(W + f"queue_{sys.argv[1]}.parquet"); b = pd.read_parquet(W + f"queue_{sys.argv[2]}.parquet")
m = a.merge(b, on=["window_id", "timestamp", "link_id"], suffixes=("_a", "_b"))
m["i"] = m.queue_pred_a & m.queue_pred_b; m["u"] = m.queue_pred_a | m.queue_pred_b
g = m.groupby("window_id")[["i", "u"]].sum(); g["agree"] = g.i / g.u.clip(lower=1); g = g.reset_index()
cond = pd.concat([pd.read_csv(REL / "task2" / p / s / "window_index.csv")[["window_id", "condition", "forecast_origin"]]
                  for p in T2_PANELS for s in ("validation", "private")])
g = g.merge(cond, on="window_id"); g["day"] = pd.to_datetime(g.forecast_origin).dt.day.clip(upper=5)
g["split"] = np.where(g.window_id.str.contains("validation"), "val", "pri")
print(g.groupby(["split", "condition"]).agree.mean().round(3).unstack())
print(g.groupby(["split", "day"]).agree.mean().round(3).unstack())
