"""Replica of the organizer Task 2 window selector (chronological, spacing across conditions).

Matches the official train windows for onset only: onset 39/40, ongoing 14/40 (audit 11/01, A-04)."""
import numpy as np
from t2_windows import queue_matrix, iou, H, F

def scan(P, Qm=None, start=H, n_each=5, spacing=72, persist_max=0.9, cov_min=0.7):
    Qm = queue_matrix(P) if Qm is None else Qm
    obs = np.isfinite(P.V) & P.E
    any_q = Qm.any(axis=1)
    on, og = [], []
    cands_og = []
    for t in range(max(start, H), P.T - F):
        if P.times[t - H][:10] != P.times[t + F][:10]: continue
        if obs[t - H:t].mean() < cov_min: continue
        if not any_q[t + 1:t + F + 1].any(): continue
        hq = Qm[t - H:t]
        est = (hq.sum() >= 2) and (hq.sum(axis=0).max() >= 2)
        if not est and not hq.any() and not any_q[t]:
            if len(on) < n_each and all(abs(t - o) >= spacing for o in on): on.append(t)
        elif est:
            if iou(np.broadcast_to(Qm[t - 1], (F, P.n)), Qm[t + 1:t + F + 1], P.E[t + 1:t + F + 1]) <= persist_max:
                cands_og.append(t)
        if len(on) >= n_each and len(cands_og) > 2000: break
    for t in cands_og:
        if len(og) >= n_each: break
        if all(abs(t - o) >= spacing for o in on + og): og.append(t)
    return on, og
