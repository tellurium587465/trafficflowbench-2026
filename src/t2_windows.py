"""Task 2 local instrument: windows on the train split that follow the organizer selector.

queue cell      : eligible, finite speed, speed <= v_cut (0.6 * links.csv free speed)
onset  origin T : no queued cell in rows T-12..T+5, at least one queued cell at T+6
ongoing origin T: history rows T-12..T-1 hold >=2 queued obs with >=2 on one link,
                  a queued cell somewhere in T+1..T+6, persistence IoU <= 0.9,
                  history coverage >= 0.7
Windows of one condition are kept >= 72 rows (6 h) apart within a panel.
"""
from __future__ import annotations
import numpy as np
from cachelib import load, T2_PANELS

H, F = 12, 6


def queue_matrix(P):
    V = P.V
    return (np.nan_to_num(V, nan=1e9) <= P.vcut[None, :]) & P.E


def iou(pred, true, elig):
    p = pred & elig; t = true & elig
    u = (p | t).sum()
    return 1.0 if u == 0 else (p & t).sum() / u


def candidates(P, rng, spacing=72):
    Qm = queue_matrix(P)
    obs = np.isfinite(P.V) & P.E
    T = P.T
    any_q = Qm.any(axis=1)
    cnt = np.concatenate([np.zeros((1, P.n), np.int32), np.cumsum(Qm, axis=0, dtype=np.int32)])
    ocnt = np.concatenate([[0], np.cumsum(obs.sum(axis=1))])
    qrow = np.concatenate([[0], np.cumsum(any_q)])
    out = {"queue_onset": [], "queue_ongoing": []}
    for t in range(H, T - F):
        # same calendar day only
        if P.times[t - H][:10] != P.times[t + F][:10]:
            continue
        cov = (ocnt[t] - ocnt[t - H]) / (H * P.n)
        if cov < 0.7:
            continue
        hq_link = cnt[t] - cnt[t - H]
        hist_q = hq_link.sum()
        fut_any = qrow[t + F + 1] - qrow[t + 1] > 0
        if not fut_any:
            continue
        if qrow[t + F] - qrow[t - H] == 0 and any_q[t + F]:
            out["queue_onset"].append(t)
        elif hist_q >= 2 and hq_link.max() >= 2:
            last = Qm[t - 1]
            fut = Qm[t + 1:t + F + 1]; el = P.E[t + 1:t + F + 1]
            if iou(np.broadcast_to(last, fut.shape), fut, el) <= 0.9:
                out["queue_ongoing"].append(t)
    sel = {}
    for cond, ts in out.items():
        chosen = []
        for t in rng.permutation(ts):
            if all(abs(t - o) >= spacing for o in chosen):
                chosen.append(int(t))
        sel[cond] = sorted(chosen)
    return sel, Qm


def build(spacing=72, seed=0):
    rng = np.random.default_rng(seed)
    W = []
    for panel in T2_PANELS:
        P = load(panel, "train")
        sel, Qm = candidates(P, rng, spacing)
        for cond, ts in sel.items():
            for t in ts:
                W.append(dict(panel=panel, t=t, cond=cond, date=P.times[t][:10],
                              label=Qm[t + 1:t + F + 1].copy(), elig=P.E[t + 1:t + F + 1].copy()))
        print(panel, {k: len(v) for k, v in sel.items()}, flush=True)
    return W


def score(W, preds):
    """Organizer aggregation: window -> (panel, cond) -> panel -> family -> mean."""
    from collections import defaultdict
    pc = defaultdict(list)
    for w, p in zip(W, preds):
        pc[(w["panel"], w["cond"])].append(iou(p.astype(bool), w["label"], w["elig"]))
    panels = sorted({k[0] for k in pc})
    pm = {p: np.mean([np.mean(pc[(p, c)]) for c in ("queue_onset", "queue_ongoing") if (p, c) in pc]) for p in panels}
    fam = defaultdict(list)
    for p, v in pm.items():
        fam[p.rsplit("_", 1)[0]].append(v)
    on = np.mean([np.mean(pc[(p, "queue_onset")]) for p in panels if (p, "queue_onset") in pc])
    og = np.mean([np.mean(pc[(p, "queue_ongoing")]) for p in panels if (p, "queue_ongoing") in pc])
    return float(np.mean([np.mean(v) for v in fam.values()])), float(on), float(og)
