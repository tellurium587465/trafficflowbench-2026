"""Task 1 'boundary' targets: a specialist for cells whose released neighbourhood (+-2 links, +-2 steps) holds both
congested (v < 0.6 vf) and free (v >= 0.6 vf) observations (outside the Task 2 blanks).  Audit c2/d0: these are 3% of
the non-blank targets but 33% of the speed squared error of the best Task 1.

Features = the f3 grid features (minus location identifiers pos/_vmed/_qratio) + the raw released neighbourhood
(v/vf at +-3 links x +-2 steps, q/cap at +-1 x +-1), congested/free counts, distances to the nearest congested and
free observation up/downstream at t-1, t, t+1, the side of the queue, and backward-wave (w) / free-flow (vf)
characteristic look-ups.  Target = residual to the f3 base (v/vf and q/lane), as f3.
"""
from __future__ import annotations
import numpy as np

R_SEL, S_SEL = 2, 2
DROP = ("pos", "_vmed", "_qratio")


def released(P, V, Q):
    """V, Q as a participant sees them (blank = NaN); train: P.blank masked; test: cache already masked."""
    V = V.astype(np.float32).copy(); Q = Q.astype(np.float32).copy()
    if getattr(P, "split", "") == "train" or getattr(P, "_sim", False):
        V[P.blank] = np.nan; Q[P.blank] = np.nan
    ok = np.isfinite(V) & np.isfinite(Q); V[~ok] = np.nan; Q[~ok] = np.nan
    return V, Q


def _sh(A, dt, dj):
    """value at (t+dt, j+dj) placed at (t, j); NaN outside."""
    out = np.full_like(A, np.nan, dtype=np.float32)
    T, n = A.shape
    ts, te = max(0, -dt), min(T, T - dt); js, je = max(0, -dj), min(n, n - dj)
    if ts < te and js < je:
        out[ts:te, js:je] = A[ts + dt:te + dt, js + dj:je + dj]
    return out


def selector(P, V):
    vf = P.vf[None, :].astype(np.float32)
    c = np.where(np.isfinite(V), V < 0.6 * vf, False); f = np.where(np.isfinite(V), V >= 0.6 * vf, False)
    cc = np.zeros(V.shape, np.int16); ff = np.zeros(V.shape, np.int16)
    for dj in range(-R_SEL, R_SEL + 1):
        for dt in range(-S_SEL, S_SEL + 1):
            if dj or dt:
                cc += np.nan_to_num(_sh(c.astype(np.float32), dt, dj)).astype(np.int16)
                ff += np.nan_to_num(_sh(f.astype(np.float32), dt, dj)).astype(np.int16)
    return (cc > 0) & (ff > 0), cc, ff


def extra_features(P, V, Q, r, c, cc, ff):
    vf, cap = P.vf.astype(np.float32), P.cap.astype(np.float32)
    vr = V / vf[None, :]; qc = Q / cap[None, :]
    x = np.concatenate([[0.0], np.cumsum(P.length[:-1])]).astype(np.float32)
    f = {}
    for dj in range(-3, 4):
        for dt in range(-2, 3):
            if dj or dt:
                f[f"v_{dj}_{dt}"] = _sh(vr, dt, dj)[r, c]
    for dj in (-1, 0, 1):
        for dt in (-1, 0, 1):
            if dj or dt:
                f[f"q_{dj}_{dt}"] = _sh(qc, dt, dj)[r, c]
    f["n_cong"] = cc[r, c].astype(np.float32); f["n_free"] = ff[r, c].astype(np.float32)
    cong = np.where(np.isfinite(V), V < 0.6 * vf[None, :], False); free = np.where(np.isfinite(V), V >= 0.6 * vf[None, :], False)
    for dt in (-1, 0, 1):
        for nm, M in (("c", cong), ("f", free)):
            Ms = np.nan_to_num(_sh(M.astype(np.float32), dt, 0)) > 0
            up = np.full(len(r), 9.0, np.float32); dn = np.full(len(r), 9.0, np.float32)
            for k in range(5, 0, -1):
                hu = np.zeros(len(r), bool); hd = np.zeros(len(r), bool)
                okj = c - k >= 0; hu[okj] = Ms[r[okj], c[okj] - k]
                okj = c + k < P.n; hd[okj] = Ms[r[okj], c[okj] + k]
                up[hu] = k; dn[hd] = k
            f[f"d{nm}_up_{dt}"] = up; f[f"d{nm}_dn_{dt}"] = dn
    f["queue_down"] = (f["dc_dn_0"] < f["dc_up_0"]).astype(np.float32)          # congestion on the downstream side -> tail region
    # characteristic look-ups: backward wave (congested info travels upstream at w), free flow (downstream at vf)
    w = float(np.nanmedian(P.w)); vff = float(np.nanmedian(P.vf))
    for k in (1, 2):
        dtk = 5.0 / 60 * k
        for nm, sp, sign in (("cw", w, +1), ("ff", vff, -1)):
            pos = x[c] + sign * sp * dtk                                          # source position k steps earlier
            j = np.clip(np.searchsorted(x, pos), 0, P.n - 1)
            src = _sh(vr, -k, 0)                                                   # value k steps earlier at the same link column
            f[f"{nm}_v_m{k}"] = src[r, j]
            src2 = _sh(vr, k, 0)                                                   # k steps later, mirrored characteristic
            j2 = np.clip(np.searchsorted(x, x[c] - sign * sp * dtk), 0, P.n - 1)
            f[f"{nm}_v_p{k}"] = src2[r, j2]
    return f


def build(P, F, V, Q, r, c, cc, ff, rows_fn):
    X0, names0 = rows_fn(P, F, r, c)
    keep = [i for i, n in enumerate(names0) if n not in DROP]
    ex = extra_features(P, V, Q, r, c, cc, ff)
    en = sorted(ex)
    X = np.column_stack([X0[:, keep]] + [ex[k] for k in en]).astype(np.float32)
    return X, [names0[i] for i in keep] + en


PARAMS = dict(objective="regression", learning_rate=0.05, num_leaves=127, min_data_in_leaf=100, feature_fraction=0.7,
              bagging_fraction=0.8, bagging_freq=1, lambda_l2=2.0, num_threads=8, verbose=-1, seed=5, force_row_wise=True,
              deterministic=True)
