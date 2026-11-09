"""Arrival features v2: per-horizon arrival/downstream-min-capacity ratios, cumulative excess demand over the
next 30 min, upstream demand growth, and the same quantities for the tightest section within 1 and 5 km."""
import numpy as np
from t2_arrival import arrival_features, _dmin

_DM2 = {}


def _dmin_r(P, x, r):
    key = (P.panel, r)
    if key not in _DM2:
        _DM2[key] = np.array([P.cap[(x >= x[i]) & (x <= x[i] + r)].min() for i in range(P.n)])
    return _DM2[key]


def arrival_features2(P, Vwin, Qwin, tl, x):
    X1, n1 = arrival_features(P, Vwin, Qwin, tl, x)
    arr = {h: X1[:, n1.index(f"arr{h}_c")] * P.cap for h in (10, 20, 30, 35)}
    feats, names = [X1], list(n1)
    for r in (1.0, 5.0):
        dm = _dmin_r(P, x, r)
        for h in (10, 20, 30, 35):
            feats.append((arr[h] / dm)[:, None]); names.append(f"arr{h}_dn{int(r)}")
        exc = sum(np.maximum(arr[h] - dm, 0) for h in (10, 20, 30, 35)) / dm
        feats.append(exc[:, None]); names.append(f"excess_dn{int(r)}")
    # upstream demand growth: arrival-equivalent flow 30 min earlier (row tl-7) vs now (row tl-1)
    last = tl - 1; prev = max(tl - 7, 0)
    q_prev = Qwin[prev]; ok = np.isfinite(q_prev)
    vf = float(np.nanmedian(P.vf))
    if ok.sum() >= 2:
        a_prev = np.interp(x - vf * 35 / 60, x[ok], q_prev[ok], left=np.nan, right=np.nan)
    else:
        a_prev = np.full(P.n, np.nan)
    feats.append(((arr[35] - a_prev) / P.cap)[:, None]); names.append("arr35_growth")
    return np.column_stack(feats).astype(np.float32), names
