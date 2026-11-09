"""Timing shift of congestion: per panel, the distribution of the hour of queued observed cells (weekdays, observed
layer: finite and not a Task 1 blank on train; finite on validation/private = their released layer). Real train
months pairwise vs real test months vs train (diagnostic of the generator, nothing is fitted on it), and twin world A
months pairwise vs twin test worlds vs A. Statistic: mean |difference| of the queued-cell hour histograms (TV over
24 bins) and the shift of the mean hour."""
import sys, numpy as np, pandas as pd
sys.path.insert(0, r"C:/Users/user/tfb/src")
from cachelib import load, T2_PANELS


def hist(P, train_mask):
    q = (np.nan_to_num(P.V, nan=1e9) <= P.vcut[None, :]) & np.isfinite(P.V) & P.E
    if train_mask:
        q &= ~P.blank
    hr = np.array([int(str(t)[11:13]) for t in P.times]); wk = np.array([pd.Timestamp(str(t)[:10]).dayofweek < 5 for t in P.times[::288]]).repeat(288)[:P.T]
    mon = np.array([str(t)[:7] for t in P.times]); out = {}
    for m in np.unique(mon):
        s = (mon == m) & wk; c = np.bincount(hr[s], weights=q[s].sum(1), minlength=24)
        if c.sum() > 50:
            out[m] = c / c.sum()
    return out


tv = lambda a, b: 0.5 * np.abs(a - b).sum(); mh = lambda h: (h * np.arange(24)).sum()
rows = []
for p in T2_PANELS:
    R = hist(load(p, "train"), True); Rm = np.mean(list(R.values()), axis=0)
    rr = np.mean([tv(R[a], R[b]) for i, a in enumerate(R) for b in list(R)[i + 1:]])
    te = [hist(load(p, s), False) for s in ("validation", "private")]
    rt = [tv(list(h.values())[0], Rm) for h in te if h]; rs = [mh(list(h.values())[0]) - mh(Rm) for h in te if h]
    A = hist(load(p, "twinA"), True); Am = np.mean(list(A.values()), axis=0)
    aa = np.mean([tv(A[a], A[b]) for i, a in enumerate(A) for b in list(A)[i + 1:]])
    tw = [hist(load(p, f"twinT{k}"), True) for k in range(8)]
    at = [tv(list(h.values())[0], Am) for h in tw if h]; ash = [mh(list(h.values())[0]) - mh(Am) for h in tw if h]
    rows.append((p, rr, np.mean(rt), np.mean(np.abs(rs)), aa, np.mean(at), np.mean(np.abs(ash))))
D = pd.DataFrame(rows, columns=["panel", "real_train_pair", "real_test_vs_train", "real_test_|dhour|", "twinA_pair", "twin_test_vs_A", "twin_test_|dhour|"])
print(D.round(3).to_string(index=False)); print(D.mean(numeric_only=True).round(3).to_dict())
