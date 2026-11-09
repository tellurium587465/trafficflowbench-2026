"""G34: is breakdown deterministic per site (generator identification premise)? Train months, every day, morning
first queue (05:00-12:00; first row with a queued link (v < vcut) on two consecutive rows). b = the link just
downstream of the most downstream queued link. Breakdown flow = max over links b..b+2 of the mean flow over the 30 min
before the first queue row. Per site (panel, b): count, median breakdown flow / fd capacity, CV across days.
Also, on the same days, the 'challenge' at the other regular sites: their own pre-breakdown flow / their median
breakdown flow (did a site carry its usual breakdown flow without breaking?).
usage: python g34_breakdown_flow.py"""
import sys, numpy as np, pandas as pd, warnings
warnings.simplefilter("ignore")
sys.path.insert(0, r"C:/Users/user/tfb/src")
from cachelib import load, T2_PANELS
ev = []
for p in T2_PANELS:
    P = load(p, "train"); D = P.T // 288
    qv = (P.V <= P.vcut[None, :]) & np.isfinite(P.V)
    for d in range(D):
        base = d * 288
        for s in range(60, 144):
            r = base + s
            both = qv[r] & qv[r + 1]
            if both.any():
                b = min(int(np.flatnonzero(both).max()) + 1, P.n - 1)
                fl = np.nanmean(P.Q[r - 6:r, b:b + 3], axis=0)
                bf = np.nanmax(fl) if np.isfinite(fl).any() else np.nan
                ev.append(dict(panel=p, day=d, slot=s, b=b, bf=bf, cap=P.cap[b], lanes=P.lanes[b],
                               we=pd.Timestamp(str(P.day[d])).dayofweek >= 5))
                break
E = pd.DataFrame(ev); E["ratio"] = E.bf / E.cap
print(f"days with a morning first queue: {len(E)}")
G = E.groupby(["panel", "b"]).agg(n=("bf", "size"), bf_med=("bf", "median"), cv=("bf", lambda x: np.nanstd(x) / np.nanmean(x)),
                                   ratio=("ratio", "median"), slot_sd=("slot", "std"))
G = G[G.n >= 8].sort_values("n", ascending=False)
print(G.round(3).head(30).to_string())
print(f"\nsites with n>=8: {len(G)}; weighted median CV of breakdown flow {np.average(G.cv, weights=G.n):.3f}")
E.to_csv(r"C:/Users/user/tfb/work/t2/g34_events.csv", index=False)
