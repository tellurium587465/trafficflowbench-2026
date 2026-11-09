"""G17b: is the congested branch of the FD redrawn in the test months (as vf is, F3)? Triangular FD, congested side
q = w (kj - q/v)  =>  per congested cell (v < 0.5 vf, q > 0) the implied  kj_eff = q / w_file + q / v  (w from
fd_parameters.csv). Per link, median over days 1-4 of each month (links with >= 30 congested cells); log-ratio to the
median of the other train months. Also the backward-wave speed alone: slope of q against k = q/v by least squares
through (kj_file, 0) is not identifiable without kj, so only kj_eff is reported."""
import sys, numpy as np, pandas as pd
sys.path.insert(0, r"C:/Users/user/tfb/src")
from cachelib import load, PANELS

out = []
for p in PANELS:
    PT = load(p, "train"); mons = sorted(set(t[:7] for t in PT.times))
    def stat(P, rows):
        V, Q = P.V[rows], P.Q[rows]
        ok = P.E[rows] & np.isfinite(V) & np.isfinite(Q) & (V < 0.5 * P.vf[None, :]) & (V > 3) & (Q > 50)
        kj = np.where(ok, Q / P.w[None, :] + Q / np.maximum(V, 1e-3), np.nan) / P.lanes[None, :]
        n = ok.sum(0)
        with np.errstate(all="ignore"):
            m = np.nanmedian(kj, axis=0)
        m[n < 30] = np.nan
        return m
    def rows_of(P, mon):
        return np.array([t[:7] == mon and int(t[8:10]) <= 4 for t in P.times])
    S = {m: stat(PT, rows_of(PT, m)) for m in mons}
    for sp in ("validation", "private"):
        P = load(p, sp); m = P.times[0][:7]; S[m] = stat(P, rows_of(P, m))
    for m, k in S.items():
        others = [o for o in mons if o != m]
        with np.errstate(all="ignore"):
            r = np.log(k / np.nanmedian(np.stack([S[o] for o in others]), axis=0))
        n = np.isfinite(r).sum()
        if n >= 3:
            out.append((p, m, "test" if m not in mons else "train", n, np.nanstd(r), np.nanmedian(np.abs(r)), np.nanmedian(r)))
R = pd.DataFrame(out, columns=["panel", "month", "kind", "nlinks", "sd", "mad", "med"])
print(R.groupby("kind")[["nlinks", "sd", "mad", "med"]].mean().round(4))
print(R[R.kind == "test"].round(4).to_string(index=False))
