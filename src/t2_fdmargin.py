"""Location-free physics features for T2 (traffic review P1), from the history only (rows <= T-5).

  q_star   flow at which the congested FD branch has speed 0.6 vf: q* = w kj / (1 + w / (0.6 vf))
  band_q   median flow over the last 3 rows within the near-congested band (contiguous links with v < 0.95 vf
           around the link; NaN if the link is not in a band), and its slope over T-30..T-5
  margin   band_q / q* - 1  (inside a band, links with margin < 0 sit below the queue threshold)
  margin30 margin extrapolated 30 min with the band slope
  surplus  link flow / median flow of links +-3..8 away (hidden ramps), and its change over 30 min
"""
import numpy as np

NAMES = ["fd_qstar_c", "band_q_c", "fd_margin", "fd_margin30", "q_surplus", "q_surplus_d30"]


def _band_ids(slow):
    """label contiguous runs of True with ids 1..; 0 = not in a band."""
    ids = np.zeros(len(slow), int); cur = 0; prev = False
    for i, s in enumerate(slow):
        if s and not prev:
            cur += 1
        ids[i] = cur if s else 0; prev = s
    return ids


_OFF = np.r_[np.arange(-8, -2), np.arange(3, 9)]


def _surplus(q):
    """q_i / median(q_{i-8..i-3}, q_{i+3..i+8}) (vectorised; needs >= 3 finite neighbours)."""
    n = len(q); qp = np.r_[np.full(8, np.nan), q, np.full(8, np.nan)]
    nb = qp[np.arange(n)[:, None] + 8 + _OFF[None, :]]
    cnt = np.isfinite(nb).sum(axis=1)
    with np.errstate(all="ignore"):
        med = np.nanmedian(np.where(cnt[:, None] >= 3, nb, np.nan), axis=1)
    return q / np.maximum(med, 1.0)


def fdmargin_features(P, V, Q, tl):
    vf, w, kj = P.vf, P.w, P.kj
    qstar = w * kj / (1 + w / (0.6 * vf))
    last = [r for r in (tl - 1, tl - 2, tl - 3) if r >= 0]
    v = np.nanmean(np.stack([V[r] for r in last]), axis=0) if last else np.full(P.n, np.nan)
    q3 = np.stack([Q[r] for r in last]) if last else np.full((1, P.n), np.nan)
    slow = np.isfinite(v) & (v < 0.95 * vf)
    ids = _band_ids(slow)
    band_q = np.full(P.n, np.nan); band_q_old = np.full(P.n, np.nan)
    r_old = [r for r in (tl - 6, tl - 7, tl - 8) if r >= 0]
    for b in range(1, ids.max() + 1):
        m = ids == b
        band_q[m] = np.nanmedian(q3[:, m]) if np.isfinite(q3[:, m]).any() else np.nan
        if r_old:
            qo = np.stack([Q[r] for r in r_old])[:, m]
            band_q_old[m] = np.nanmedian(qo) if np.isfinite(qo).any() else np.nan
    margin = band_q / qstar - 1
    slope = (band_q - band_q_old) / 5.0                          # per 5-min step over ~25 min
    margin30 = (band_q + 6 * np.nan_to_num(slope)) / qstar - 1
    qn = np.nanmean(q3, axis=0); s_now = _surplus(qn)
    s_old = _surplus(np.nanmean(np.stack([Q[r] for r in r_old]), axis=0)) if r_old else np.full(P.n, np.nan)
    return np.column_stack([qstar / P.cap, band_q / P.cap, margin, margin30, s_now, s_now - s_old]).astype(np.float32)
