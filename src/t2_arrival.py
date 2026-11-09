"""Arrival-flow features: flow that, travelling at free-flow speed, reaches link i at T+5k is now
(at the last history row) at x_i - vf*(5k+5)/60 km upstream. Upstream = lower index (all panels)."""
import numpy as np

H_MIN = (10, 20, 30, 35)
_DM = {}


def _dmin(P, x):
    key = (P.panel, P.n)
    if key not in _DM:
        _DM[key] = np.array([P.cap[(x >= x[i]) & (x <= x[i] + 3.0)].min() for i in range(P.n)])
    return _DM[key]


def arrival_features(P, Vwin, Qwin, tl, x):
    """Vwin/Qwin: causal day arrays (rows up to the window's last usable row tl-? handled by caller).
    Uses the last history row (tl-1, i.e. T-5) and the row 30 min earlier for trends."""
    last = tl - 1
    q_now = Qwin[last]
    ok = np.isfinite(q_now)
    xs = x[ok]; qs = q_now[ok]
    vf = float(np.nanmedian(P.vf))
    feats = []
    names = []
    # boundary (most upstream observed) flow level and trend over the last 30 min
    b_now = np.nanmean(Qwin[max(last - 2, 0):last + 1, :3])
    b_prev = np.nanmean(Qwin[max(last - 8, 0):max(last - 5, 1), :3])
    for h in H_MIN:
        pos = x - vf * h / 60.0
        if len(xs) >= 2:
            arr = np.interp(pos, xs, qs, left=np.nan, right=np.nan)
        else:
            arr = np.full(P.n, np.nan)
        beyond = pos < x[0]
        extra = np.where(beyond, b_now + (b_now - b_prev) * np.minimum(-pos / (vf * 30 / 60), 2.0), np.nan)
        arr = np.where(beyond, extra, arr)
        feats.append(arr / P.cap); names.append(f"arr{h}_c")
    feats.append(np.full(P.n, (b_now - b_prev) / max(np.nanmean(P.cap[:3]), 1))); names.append("bnd_trend")
    # downstream-min capacity within 3 km (where the arriving flow will meet the tightest section)
    dmin = _dmin(P, x)
    feats.append(dmin / P.cap); names.append("cap_dn_min_rel")
    feats.append(feats[2] * P.cap / dmin); names.append("arr30_over_dnmin")
    return np.column_stack(feats).astype(np.float32), names
