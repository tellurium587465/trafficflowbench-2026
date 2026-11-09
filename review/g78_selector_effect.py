"""G78 (INSTRUMENT_V3_SPEC §2.2, 10/23): the organisers' selector vs our pseudo-window rules, with answers (train split, June
2030). Official = the 80 train-split windows (window_index.csv); pseudo = our selector-like windows of June (cache is_sel).
Structural statistics for ALL windows (no model): hour, weekend, links queued at T-1, persistence IoU (queue at T vs
T+30), share of the T+30 queued links that are rare (f from the OTHER train months, panel x weekend x 2-hour slot,
f <= 0.1). Model statistics where a month-CV OOF exists (q25_nomem seeds 3,4: June held out): IoU of the expected-IoU
set (onset: T+30 row, ongoing: all rows; eligible cells; no evidence rule), confidence = max probability at T+30.
Official windows are matched to cache windows by (panel, origin row); 53 / 80 coincide with our selector replica.
usage: python g78_selector_effect.py"""
import sys, os, pickle
os.environ["TFB_T2CACHE"] = "2_noT_af"
sys.path.insert(0, r"C:/Users/user/tfb/src")
sys.argv = [sys.argv[0], "0"]
import numpy as np, pandas as pd
from t2_eval_oof import T
from cachelib import load, T2_PANELS
from common import REL
from t2_model import best_set
from t2_windows import queue_matrix

W = "C:/Users/user/tfb/work/t2/"
OOF = [pickle.load(open(W + f"oof_q25_nomem_s{s}.pkl", "rb")) for s in (3, 4)]
mon = np.array([str(d)[:7] for d in T.META["date"]])
key = lambda p, we, t: (str(p), bool(we), (int(t) % 288) // 24)
CNT, DEN = {}, {}                                             # rarity from the train months except June
for k in range(T.NWIN):
    if T.META["is_sel"][k] and str(T.META["cond"][k]) == "queue_onset" and mon[k] != "2030-06":
        lab, el = T.LAB[k]; kk = key(T.META["panel"][k], pd.Timestamp(str(T.META["date"][k])[:10]).dayofweek >= 5, T.META["t"][k])
        CNT[kk] = CNT.get(kk, 0) + (lab[5] & el[5]).astype(float); DEN[kk] = DEN.get(kk, 0) + 1
idx = {(str(p), int(t)): k for k, (p, t) in enumerate(zip(T.META["panel"], T.META["t"]))}
QM = {}


def stats(p, t, cond, k):
    if p not in QM:
        P = load(p, "train"); QM[p] = (P, queue_matrix(P))
    P, Q = QM[p]; we = pd.Timestamp(P.times[t][:10]).dayofweek >= 5
    q30 = Q[t + 6].astype(bool); qT = Q[t].astype(bool); qm1 = Q[t - 1].astype(bool)
    f = CNT[key(p, we, t)] / DEN[key(p, we, t)] if key(p, we, t) in DEN else np.zeros(P.n)
    u = (qT | q30).sum()
    r = dict(cond=cond, hour=(t % 288) // 12, weekend=we, q_tm1=int(qm1.sum()), persist=float((qT & q30).sum() / u) if u else 1.0,
             rare_share=float((f[q30] <= 0.1).mean()) if q30.any() else np.nan, n_q30=int(q30.sum()))
    if k is not None and k in OOF[0]:
        lab, el = T.LAB[k]; pr = (OOF[0][k] + OOF[1][k]) / 2
        if cond == "queue_onset":
            sel = best_set(pr[5], el[5]); y, e = lab[5].astype(bool), el[5].astype(bool)
        else:
            sel = best_set(pr.ravel(), el.ravel()).reshape(pr.shape); y, e = lab.astype(bool), el.astype(bool)
        uu = ((sel | y) & e).sum(); r["iou"] = 1.0 if uu == 0 else float((sel & y & e).sum() / uu); r["conf"] = float(pr[5].max())
    return r


rows = []
for p in T2_PANELS:
    P = load(p, "train"); ti = {s: i for i, s in enumerate(P.times)}
    for x in pd.read_csv(REL / "task2" / p / "train" / "window_index.csv").itertuples():
        t = ti[x.forecast_origin]; rows.append(dict(group="official", **stats(p, t, x.condition, idx.get((p, t)))))
for k in np.flatnonzero(np.asarray(T.META["is_sel"], bool) & (mon == "2030-06")):
    rows.append(dict(group="pseudo", **stats(str(T.META["panel"][k]), int(T.META["t"][k]), str(T.META["cond"][k]), int(k))))
R = pd.DataFrame(rows); R.to_csv(W + "g78_selector_effect.csv", index=False)
for cond in ("queue_onset", "queue_ongoing"):
    print(f"== {cond}")
    g = R[R.cond == cond]
    print(g.groupby("group").agg(n=("hour", "size"), hour=("hour", "mean"), weekend=("weekend", "mean"), q_tm1=("q_tm1", "mean"),
                                 persist=("persist", "mean"), rare_share=("rare_share", "mean"), n_q30=("n_q30", "mean"),
                                 n_model=("iou", "count"), iou=("iou", "mean"), conf=("conf", "mean")).round(3).to_string())
    for h0, h1 in ((0, 9), (9, 12), (12, 15), (15, 24)):
        s = g[(g.hour >= h0) & (g.hour < h1)].groupby("group").size().to_dict(); print(f"  hours {h0}-{h1}: {s}")
