"""G32: the organizer's window-policy note says 'an hour before a queue forms the road is already at 1.28x its queue
threshold against 1.54x where none follows, and carries 6912 vph against 4620'. G22 measured 1.57x at T-5 on the train
months. The train months were not regenerated in the 9/12 rebuild, validation / private were. Compare the visible
history (T-60..T-5, allowed input) of the official onset / ongoing windows across splits:
  minr  = min over links and history rows of v / vcut      (observed cells)
  minr6 = min over links of the mean over the last 6 rows of v / vcut
  qmax  = max over links of the mean flow over the history
usage: python g32_precursor_by_split.py"""
import sys, numpy as np, pandas as pd, warnings
warnings.simplefilter("ignore")
sys.path.insert(0, r"C:/Users/user/tfb/src")
from cachelib import load, T2_PANELS
from common import REL
rows = []
for sp in ("train", "validation", "private"):
    for p in T2_PANELS:
        P = load(p, sp); ti = {s: i for i, s in enumerate(P.times)}
        R = P.V / P.vcut[None, :]
        for r in pd.read_csv(REL / "task2" / p / sp / "window_index.csv").itertuples():
            t = ti.get(r.forecast_origin)
            if t is None or t < 12:
                continue
            h = R[t - 12:t]; q = P.Q[t - 12:t]
            rows.append(dict(split=sp, panel=p, cond=r.condition, minr=np.nanmin(h), minr6=np.nanmin(np.nanmean(h[-6:], axis=0)),
                             qmax=np.nanmax(np.nanmean(q, axis=0)), hour=int(r.forecast_origin[11:13])))
D = pd.DataFrame(rows)
print(D.groupby(["split", "cond"])[["minr", "minr6", "qmax", "hour"]].agg(["median", "count"]).round(3).to_string())
