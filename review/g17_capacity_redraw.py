"""G17: are link capacities redrawn in the test months like the free-flow speed (F3)? Per link, the 99.5th percentile
of observed flow per lane over days 1-4 (finite, eligible) in each month; log-ratio to the 8-month median of the
train months. Spread between train months (each vs the others) vs test months. Also the free-flow speed (median
v over v > 0.9 vf) the same way, as the reference effect F3."""
import sys, numpy as np, pandas as pd
sys.path.insert(0, r"C:/Users/user/tfb/src")
from cachelib import load, PANELS

out = []
for p in PANELS:
    PT = load(p, "train"); mons = sorted(set(t[:7] for t in PT.times))
    def stat(P, rows):
        Q = np.where(P.E[rows] & np.isfinite(P.Q[rows]), P.Q[rows], np.nan) / P.lanes[None, :]
        V = np.where(P.E[rows] & np.isfinite(P.V[rows]) & (P.V[rows] > 0.9 * P.vf[None, :]), P.V[rows], np.nan)
        with np.errstate(all="ignore"):
            return np.nanpercentile(Q, 99.5, axis=0), np.nanmedian(V, axis=0)
    def rows_of(P, mon):
        return np.array([t[:7] == mon and int(t[8:10]) <= 4 for t in P.times])
    S = {m: stat(PT, rows_of(PT, m)) for m in mons}
    for sp in ("validation", "private"):
        P = load(p, sp); m = P.times[0][:7]; S[m] = stat(P, rows_of(P, m))
    trm = mons
    for m, (q, v) in S.items():
        others = [o for o in trm if o != m]
        qr = np.log(q / np.nanmedian(np.stack([S[o][0] for o in others]), axis=0))
        vr = np.log(v / np.nanmedian(np.stack([S[o][1] for o in others]), axis=0))
        out.append((p, m, "test" if m not in trm else "train", np.nanstd(qr), np.nanmedian(np.abs(qr)), np.nanstd(vr), np.nanmedian(np.abs(vr))))
R = pd.DataFrame(out, columns=["panel", "month", "kind", "cap_sd", "cap_mad", "vf_sd", "vf_mad"])
print(R.groupby("kind")[["cap_sd", "cap_mad", "vf_sd", "vf_mad"]].mean().round(4))
print(R[R.kind == "test"].groupby("month")[["cap_sd", "cap_mad", "vf_sd", "vf_mad"]].mean().round(4))
