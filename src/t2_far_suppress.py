"""Remove ongoing-window queue predictions far from any queue visible at T (10/09, s34: LB +0.0031, ongoing +0.021,
while the same operation costs -0.044 on the train months). In the test months the queue sites move, so "a new queue
starts at another bottleneck" predictions come from location memory and miss. D = 3 links was fixed before the probe;
it is not tuned on the LB (40 windows).

usage: python t2_far_suppress.py <in_queue_parquet> <out_queue_parquet> [D=3] [days=all|weekday|weekend]
(days splits the LB effect: the validation windows are 35-40% weekend, the private ones 17.5%)
"""
import sys, numpy as np, pandas as pd
from common import NN                                        # 10/22: data root off the failing E: (TFB_NN)

src, dst = sys.argv[1], sys.argv[2]; D = int(sys.argv[3]) if len(sys.argv) > 3 else 3
DAYS = sys.argv[4] if len(sys.argv) > 4 else "all"
names = list(np.load(f"{NN}/nn_data_afdem/names.npy", allow_pickle=True)); ci = names.index("last_queue")
d = f"{NN}/nn_test_afdem/"
X = np.load(d + "X.npy", mmap_mode="r"); off = np.load(d + "off.npy")
wid = np.load(d + "window_id.npy", allow_pickle=True).astype(str); cond = np.load(d + "cond.npy", allow_pickle=True).astype(str)
q = pd.read_parquet(src); g = dict(tuple(q.groupby("window_id", sort=False)))
assert set(g) == set(wid), "queue windows differ from the test windows"
drop = np.zeros(len(q), bool)
for i, w in enumerate(wid):
    if cond[i] != "queue_ongoing":
        continue
    sub = g[w]; ts = sorted(sub.timestamp.unique()); links = sub[sub.timestamp == ts[0]].link_id.to_numpy()
    wkend = pd.Timestamp(ts[0]).dayofweek >= 5
    if (DAYS == "weekday" and wkend) or (DAYS == "weekend" and not wkend):
        continue
    assert len(links) == off[i + 1] - off[i], ("link count", w)   # parquet link order == X row order (checked 10/09)
    lq = np.asarray(X[off[i]:off[i + 1], ci], dtype=float) == 1; near = np.zeros(len(links), bool)
    for j in np.flatnonzero(lq):
        near[max(0, j - D):j + D + 1] = True
    far = set(links[~near])
    drop[sub.index[sub.link_id.isin(far).to_numpy() & (sub.queue_pred.to_numpy() == 1)]] = True
q.loc[drop, "queue_pred"] = 0
q.to_parquet(dst, index=False)
print(f"D={D} days={DAYS}: removed {drop.sum()} far ongoing positives of {int(q.queue_pred.sum()) + drop.sum()} -> {dst}")
