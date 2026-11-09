"""G221 (11/03, ledger #189 stage 0 planning): how many X1 fine-tune windows a denser pool gives on March, counts only.
The current pool (twin/build_march.py): origins every SP = 6 rows per condition, history coverage >= 0.5, a queue in
T+1..T+6, onset = no queue in T-12..T+5 and one at T+6, ongoing = >= 2 queued history cells on one link and persistence
IoU <= 0.9. Variants: SP 6 / 3 / 1, with and without the ongoing conditions (persistence and history count).
usage: py -B -X utf8 g221_dense_count.py [split=validation]"""
import os, sys, numpy as np, pandas as pd
os.environ["TFB_T2_NOT"] = "1"; os.environ["TFB_T2_ALLFIN"] = "1"
sys.path.insert(0, r"C:/Users/user/tfb/src")
from cachelib import load, T2_PANELS
from common import REL
from t2_windows import queue_matrix

H, F, COV = 12, 6, 0.5
SPLIT = sys.argv[1] if len(sys.argv) > 1 else "validation"
cnt_all = {}
for p in T2_PANELS:
    P = load(p, SPLIT); ti = {s: i for i, s in enumerate(P.times)}
    Qm = queue_matrix(P); obs = np.isfinite(P.V) & P.E; n = P.n
    blank = np.zeros(P.T, bool)
    for r in pd.read_csv(REL / "task2" / p / SPLIT / "window_index.csv").itertuples():
        b0 = ti[r.forecast_origin]; blank[b0 + 1:b0 + 19] = True
    any_q = Qm.any(axis=1); qrow = np.concatenate([[0], np.cumsum(any_q)])
    cnt = np.concatenate([np.zeros((1, n), np.int32), np.cumsum(Qm, axis=0, dtype=np.int32)])
    ocnt = np.concatenate([[0], np.cumsum(obs.sum(axis=1))])
    for SP in (6, 3, 1):
        for strict in (True, False):
            last = {"queue_onset": -10 ** 9, "queue_ongoing": -10 ** 9}; k = {"queue_onset": 0, "queue_ongoing": 0}
            for t in range(H, P.T - F):
                if (t - H) // 288 != (t + F) // 288 or blank[t - H:t + F + 1].any():
                    continue
                if (ocnt[t] - ocnt[t - H]) / (H * n) < COV or not (qrow[t + F + 1] - qrow[t + 1] > 0):
                    continue
                if qrow[t + F] - qrow[t - H] == 0 and any_q[t + F]:
                    c = "queue_onset"
                else:
                    if strict:
                        hq = cnt[t] - cnt[t - H]
                        if not (hq.sum() >= 2 and hq.max() >= 2):
                            continue
                        fut = Qm[t + 1:t + F + 1]; el = obs[t + 1:t + F + 1]; lastq = np.broadcast_to(Qm[t - 1], fut.shape)
                        u = ((lastq | fut) & el).sum()
                        if (1.0 if u == 0 else (lastq & fut & el).sum() / u) > 0.9:
                            continue
                    c = "queue_ongoing"
                if t - last[c] < SP:
                    continue
                last[c] = t; k[c] += 1
            a = cnt_all.setdefault((SP, strict), {"queue_onset": 0, "queue_ongoing": 0})
            for c in k:
                a[c] += k[c]
    load.cache_clear()
for (SP, strict), a in sorted(cnt_all.items(), key=lambda x: (-x[0][0], not x[0][1])):
    print(f"SP {SP} ({SP * 5} min), ongoing conditions {'kept' if strict else 'dropped'}: onset {a['queue_onset']}, ongoing {a['queue_ongoing']}, "
          f"ratio to current {(a['queue_onset'] + a['queue_ongoing']) / sum(cnt_all[(6, True)].values()):.2f}", flush=True)
