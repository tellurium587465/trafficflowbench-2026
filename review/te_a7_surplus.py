"""TE-A7: is the onset location visible as a LOCAL FLOW SURPLUS (hidden connector/OD flow) in the history?
Isolated births (link j turns congested at t0; nothing congested within +-5 links in the previous hour).
Origin T = t0-6 (birth at T+30, as in queue_onset windows). Features at T-1 (mean of rows T-3..T-1), all links:
  qcap    = Q/cap
  surplus = Q / through-flow, through-flow = median Q of links 3..8 positions away on both sides
  sur_cap = (Q - through) / cap   (excess per unit capacity)
Ranks every link of the corridor by each score; reports hit@1 / hit@3 of the birth link (+-1 link tolerance).
Also: surplus at birth links vs all links at the same times (distribution).
Usage: python te_a7_surplus.py PANEL [...]"""
import sys, numpy as np, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, r"C:/Users/user/tfb/src")
from cachelib import load


def through(Qrow, n):
    out = np.full(n, np.nan)
    for i in range(n):
        idx = [j for j in list(range(i - 8, i - 2)) + list(range(i + 3, i + 9)) if 0 <= j < n]
        v = Qrow[idx]; v = v[np.isfinite(v)]
        if v.size >= 3:
            out[i] = np.median(v)
    return out


for panel in sys.argv[1:]:
    P = load(panel, "train")
    V, Q = P.V, P.Q
    C = np.isfinite(V) & (V <= P.vcut[None, :])
    n, T = P.n, P.T
    ev = []
    for t in range(24, T - 1):
        for j in np.flatnonzero(C[t] & ~C[t - 1]):
            lo, hi = max(j - 5, 0), min(j + 6, n)
            if not C[t - 18:t, lo:hi].any() and not C[t - 18:t - 6].any():   # whole corridor queue-free in the history too
                ev.append((t, j))
    hits = {k: [0, 0] for k in ("qcap", "surplus", "sur_cap", "q_trend_cap")}
    bs, alls = [], []
    for t0, j in ev:
        T0 = t0 - 6
        q = np.nanmean(Q[T0 - 3:T0], axis=0)
        q0 = np.nanmean(Q[T0 - 12:T0 - 8], axis=0)
        th = through(q, n)
        sc = {"qcap": q / P.cap, "surplus": q / th, "sur_cap": (q - th) / P.cap, "q_trend_cap": (q - q0) / P.cap}
        for k, s in sc.items():
            s = np.where(np.isfinite(s), s, -np.inf)
            order = np.argsort(-s)
            hits[k][0] += np.any(np.abs(order[:1] - j) <= 1); hits[k][1] += np.any(np.abs(order[:3] - j) <= 1)
        bs.append(np.nanmax(sc["surplus"][max(j - 1, 0):j + 2])); alls.append(np.nanmedian(sc["surplus"]))
    m = len(ev)
    print(f"\n=== {panel}: births with a queue-free corridor history: {m}")
    for k, (h1, h3) in hits.items():
        print(f"  rank by {k:12s}: hit@1 {h1 / m:.2f}  hit@3 {h3 / m:.2f}")
    print(f"  surplus at birth (max over j-1..j+1) median {np.nanmedian(bs):.3f} p25 {np.nanpercentile(bs, 25):.3f} | corridor median {np.nanmedian(alls):.3f}")
    load.cache_clear()
