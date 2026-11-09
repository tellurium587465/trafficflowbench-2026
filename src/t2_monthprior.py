"""Month-adaptive location priors for Task 2 (causal: only days of the same calendar month before the
window's day, plus nothing after T). Train uses the masked layer (as released) so that train and
validation/private see the same kind of evidence.

Features per link:
  mp_any   fraction of earlier days this month on which the link was queued at any time
  mp_tod   fraction of earlier days this month on which the link was queued within +-1h of the window's T+30
  mp_first fraction of earlier days this month on which the link was among the first queued links of the day
  mp_days  number of earlier days this month (same for all links)
"""
from __future__ import annotations
import numpy as np
from functools import lru_cache
from cachelib import load

NAMES = ["mp_any", "mp_tod", "mp_first", "mp_days"]


@lru_cache(maxsize=None)
def daily_tables(panel, split):
    P = load(panel, split)
    V = P.V.copy()
    if split == "train":
        V[P.blank] = np.nan
    q = (np.nan_to_num(V, nan=1e9) <= P.vcut[None, :]) & P.E
    D = P.T // 288
    qd = q[:D * 288].reshape(D, 288, P.n)
    anyq = qd.any(axis=1)                                  # (D, n)
    first = np.zeros((D, P.n), bool)
    for d in range(D):
        rows = np.flatnonzero(qd[d].any(axis=1))
        if len(rows):
            first[d] = qd[d, rows[0]]
    months = np.array([s[:7] for s in P.day])
    return qd, anyq, first, months


def features(panel, split, t):
    qd, anyq, first, months = daily_tables(panel, split)
    d = t // 288
    m = months[d]
    prev = [i for i in range(d) if months[i] == m]
    n = qd.shape[2]
    if not prev:
        return np.column_stack([np.full(n, np.nan)] * 3 + [np.zeros(n)]).astype(np.float32)
    prev = np.array(prev)
    tod = (t % 288) + 6
    lo, hi = max(tod - 12, 0), min(tod + 13, 288)
    mp_tod = qd[prev, lo:hi].any(axis=1).mean(axis=0)
    return np.column_stack([anyq[prev].mean(axis=0), mp_tod, first[prev].mean(axis=0),
                            np.full(n, len(prev))]).astype(np.float32)
