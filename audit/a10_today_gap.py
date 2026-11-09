"""Audit A10: earlier Task 2 gaps on the same day / same month hide congestion in validation/private, but never in train
(the train masked layer has no gaps).  Features that read earlier rows of the day (q_today) or earlier days of the month
(t2_monthprior mp_*) therefore see systematically less congestion at inference than in training.
(1) val/private: windows with an earlier window on the same day; gap rows hidden before T-12
(2) train emulation on the selector-like runs (vlb_runs): q_today and mp_first with vs without the earlier windows' gaps."""
import sys, pickle, numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, r"C:/Users/user/tfb/src")
from cachelib import load, T2_PANELS
from common import REL
from t2_windows import queue_matrix

aff = []
for sp in ("validation", "private"):
    for p in T2_PANELS:
        wi = pd.read_csv(REL / "task2" / p / sp / "window_index.csv"); wi["T"] = pd.to_datetime(wi.forecast_origin)
        for r in wi.itertuples():
            same = wi[(wi.date == r.date) & (wi["T"] < r.T)]
            aff.append(dict(split=sp, panel=p, cond=r.condition, earlier_same_day=len(same) > 0,
                            earlier_days_hidden=int(((pd.to_datetime(wi.date) < pd.Timestamp(r.date)) & (wi.condition == "queue_onset")).sum())))
a = pd.DataFrame(aff)
print("share of scored windows with an earlier window on the same day:", a.groupby(["split", "cond"]).earlier_same_day.mean().round(3).to_dict())
print("mean number of earlier-day onset windows (their first-queue-of-day is hidden from mp_first):", a.groupby(["split", "cond"]).earlier_days_hidden.mean().round(2).to_dict())

runs = pickle.load(open("C:/Users/user/tfb/work/t2/vlb_runs2_noT.pkl", "rb"))
res = []
for p in T2_PANELS:
    P = load(p, "train"); V = P.V.copy(); V[P.blank] = np.nan
    q = (np.nan_to_num(V, nan=1e9) <= P.vcut[None, :]) & P.E
    for d0, ws in runs.items():
        ts = sorted(t for pp, t in ws if pp == p)
        for i, t in enumerate(ts):
            d = t - t % 288
            hid = np.zeros(P.T, bool)
            for t2 in ts[:i]:
                hid[t2 + 1:t2 + 19] = True
            rows = np.arange(d, max(t - 12, d))
            full = q[rows].sum(); gapped = q[rows][~hid[rows]].sum()
            if hid[rows].any():
                res.append(dict(panel=p, full=int(full), gapped=int(gapped)))
    load.cache_clear()
r = pd.DataFrame(res)
print(f"train emulation: windows whose same-day history before T-12 overlaps an earlier window's gap: {len(r)}; "
      f"q_today (queued cells before T-12) mean {r.full.mean():.1f} without gaps vs {r.gapped.mean():.1f} with gaps "
      f"(hidden share {1 - r.gapped.sum() / max(r.full.sum(), 1):.2%})")
