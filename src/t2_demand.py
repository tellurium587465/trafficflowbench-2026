"""Demand forecast features from the weekday x time-of-day flow profile (the generator's demand law).

35-min-ahead flow MAE (free flow, held-out months): persistence 7-9%, linear trend 6.5-8.6%,
profile ratio 3.2-4.2% (noise floor ~2.6%). Per link: today's level = mean over the last 3 history rows of
q / P(dow, tod); forecast at T+h = level * P(dow, tod_T + h). Profiles exclude the window's own month in
training (leave-one-month-out) and use all train months for validation/private windows.
"""
import numpy as np

HS = (10, 20, 30)
NAMES = [f"dem{h}_c" for h in HS] + ["dem30_over_dnmin", "dem_shape30", "dem_level"]
_PROF = {}


def _dow(day):
    return int(np.datetime64(day).astype("datetime64[D]").view("int64") % 7)


def profile(PT, exclude_month=None):
    """(7, 288, n) mean flow by weekday and time of day from the train panel PT, optionally leaving a month out."""
    key = (PT.panel, exclude_month)
    if key not in _PROF:
        Q = np.where(PT.E & (PT.P >= 75), PT.Q, np.nan); D = PT.T // 288
        Qd = Q[:D * 288].reshape(D, 288, PT.n)
        mon = np.array([d[:7] for d in PT.day[:D]]); dow = np.array([_dow(d) for d in PT.day[:D]])
        pr = np.full((7, 288, PT.n), np.nan)
        for w in range(7):
            s = (dow == w) & (mon != exclude_month)
            if s.any():
                with np.errstate(all="ignore"):
                    pr[w] = np.nanmean(Qd[s], axis=0)
        _PROF[key] = pr
    return _PROF[key]


def demand_features(P, Q, tl, t_abs, day, prof, dmin):
    """Q: day array up to row tl (row tl = T, unused); t_abs: absolute row of T; prof: (7, 288, n)."""
    pr = prof[_dow(day)]; s = t_abs % 288
    rows = [r for r in (tl - 1, tl - 2, tl - 3) if r >= 0]
    lv = np.nanmean(np.stack([Q[r] / np.maximum(pr[(s - (tl - r)) % 288], 1.0) for r in rows]), axis=0) if rows else np.full(P.n, np.nan)
    feats = [lv * pr[(s + h // 5) % 288] / P.cap for h in HS]
    feats.append(feats[-1] * P.cap / dmin)
    feats.append(pr[(s + 6) % 288] / np.maximum(pr[(s - 1) % 288], 1.0))
    feats.append(lv)
    return np.column_stack(feats).astype(np.float32)
