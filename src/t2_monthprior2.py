"""Month-adaptive location priors v2 (causal: earlier days of the same month only), robust to the release blanks.

Audit F2: in validation/private the horizon + 60-min buffer (rows T+5..T+90, all links) of every scored window is
blank, and onset windows are exactly the first queue of the day early in the month -> the v1 'first queue of the
day' feature is a different thing at inference. v2:
  * train tables get the same blanks: the organizer selector is run from each month start and rows t+1..t+18 of
    the chosen windows are hidden before counting;
  * 'first queue of the day' is replaced by 'queue head' (queued link whose downstream neighbour is free): heads
    sit at the bottleneck and remain visible after the blank.
Features per link: mp2_any, mp2_tod (queued within +-1h of T+30), mp2_head, mp2_days.
"""
from __future__ import annotations
import os
from functools import lru_cache
import numpy as np
from cachelib import load
from t2_windows import queue_matrix
from t2_selector import scan

NAMES = ["mp2_any", "mp2_tod", "mp2_head", "mp2_fhead", "mp2_thead", "mp2_days"]


@lru_cache(maxsize=None)
def daily_tables(panel, split):
    P = load(panel, split)
    V = P.V.copy()
    if split == "train":
        V[P.blank] = np.nan
    q = (np.nan_to_num(V, nan=1e9) <= P.vcut[None, :]) & P.E
    if split == "train":                                   # reproduce the release blanks of month-start windows
        Qm = queue_matrix(P); mon = np.array([s[:7] for s in P.times])
        for m in np.unique(mon):
            rows = np.flatnonzero(mon == m); r0, r1 = rows[0], rows[-1] + 1
            on, og = scan(P, Qm, start=r0 + 12)
            for t in on + og:
                if r0 <= t < r1:
                    q[t + 1:t + 19] = False
    head = q.copy(); head[:, :-1] &= ~q[:, 1:]             # downstream = higher link index
    D = P.T // 288
    qd = q[:D * 288].reshape(D, 288, P.n)
    hd = head[:D * 288].reshape(D, 288, P.n)
    anyq = qd.any(axis=1); headd = hd.any(axis=1)
    fhead = np.zeros((D, P.n), bool)                       # heads of the first VISIBLE queue of the day
    for d in range(D):
        r = np.flatnonzero(qd[d].any(axis=1))
        if len(r):
            fhead[d] = hd[d, r[0]]
    months = np.array([s[:7] for s in P.day])
    return qd, hd, anyq, headd, fhead, months


def features(panel, split, t):
    qd, hd, anyq, headd, fhead, months = daily_tables(panel, split)
    d = t // 288; m = months[d]
    prev = np.array([i for i in range(d) if months[i] == m], int)
    if os.environ.get("TFB_MP2_MAXDAYS"):                       # 10/10: train/test parity of evidence depth (test windows: days 1-4)
        prev = prev[-int(os.environ["TFB_MP2_MAXDAYS"]):]
    n = qd.shape[2]
    if not len(prev):
        return np.column_stack([np.full(n, np.nan)] * 5 + [np.zeros(n)]).astype(np.float32)
    tod = (t % 288) + 6
    lo, hi = max(tod - 12, 0), min(tod + 13, 288)
    return np.column_stack([anyq[prev].mean(axis=0), qd[prev, lo:hi].any(axis=1).mean(axis=0),
                            headd[prev].mean(axis=0), fhead[prev].mean(axis=0), hd[prev, lo:hi].any(axis=1).mean(axis=0),
                            np.full(n, len(prev))]).astype(np.float32)
