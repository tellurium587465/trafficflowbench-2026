"""G40: is the LB ongoing loss explained by queues at sites that are rare in the train months? Inputs only (no labels):
for each official ongoing window (train / validation / private), the queued links at T-1 (visible history) and their
train-month frequency f (share of train days with a queue at the link +-2 in slots T-6..T+11, same day type; for the
train windows the window's own month is excluded). With the local IoU by f bin (G31/G37 ongoing: <=0.1 0.757,
0.1-0.3 0.917, 0.3-0.6 0.926, 0.6-0.9 0.957, >0.9 0.980) the f mix predicts the ongoing score of each month.
usage: python g40_ongoing_shift_f.py"""
import sys, numpy as np, pandas as pd, warnings
warnings.simplefilter("ignore")
sys.path.insert(0, r"C:/Users/user/tfb/src")
from cachelib import load, T2_PANELS
from common import REL
FC = {}
g30 = open(r"C:/Users/user/tfb/review/g30_onset_priorshift.py", encoding="utf-8").read()
exec(g30[g30.index("def month_counts"):g30.index("def iou_last")])
rows = []
for sp in ("train", "validation", "private"):
    for p in T2_PANELS:
        P = load(p, sp); ti = {s: i for i, s in enumerate(P.times)}; q = (P.V <= P.vcut[None, :]) & np.isfinite(P.V)
        for r in pd.read_csv(REL / "task2" / p / sp / "window_index.csv").itertuples():
            if r.condition != "queue_ongoing":
                continue
            t = ti[r.forecast_origin]; tg = P.target.astype(bool)          # review 10/13: hide Task 1 targets in every split
            now = (q[t - 1] & ~tg[t - 1]) | (q[t - 2] & ~tg[t - 2])
            if not now.any():
                continue
            day = pd.Timestamp(r.forecast_origin[:10]); mon = r.forecast_origin[:7] if sp == "train" else "none"
            f = freq_excl(p, mon, day.dayofweek >= 5)[t % 288]
            fm = float(np.mean(f[now]))
            b = int(np.digitize(fm, [0.1, 0.3, 0.6, 0.9]))
            rows.append(dict(split=sp, panel=p, f=fm, bin=b))
R = pd.DataFrame(rows)
print(R.groupby("split").agg(n=("f", "size"), f_median=("f", "median"), share_f_le_0_1=("f", lambda x: np.mean(x <= 0.1)),
                             share_f_le_0_3=("f", lambda x: np.mean(x <= 0.3))).round(3).to_string())
