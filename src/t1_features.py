"""Task 1 features, computed on whole (time x link) grids with array ops only.

Inputs are the released observation layer as a participant sees it:
train -> unmasked values with every masked-layer blank set to NaN;
validation/private -> the masked layer as cached.
"""
from __future__ import annotations
import numpy as np
from cachelib import load

DT_H = 5.0 / 60.0
NB = 24  # max steps searched for prev/next observation


def inputs(P):
    V = P.V.astype(np.float32).copy(); Q = P.Q.astype(np.float32).copy()
    if P.split == "train":
        V[P.blank] = np.nan; Q[P.blank] = np.nan
    ok = np.isfinite(V) & np.isfinite(Q) & (P.P >= 75)
    V[~ok] = np.nan; Q[~ok] = np.nan
    return V, Q, ok


def prev_next(M, ok):
    T = M.shape[0]
    pos = np.arange(T, dtype=np.int32)[:, None]
    p = np.maximum.accumulate(np.where(ok, pos, -1), axis=0)
    n = np.minimum.accumulate(np.where(ok, pos, T)[::-1], axis=0)[::-1]
    return p, n


def take(M, idx):
    T = M.shape[0]
    cols = np.arange(M.shape[1])[None, :]
    v = M[np.clip(idx, 0, T - 1), cols]
    return np.where((idx >= 0) & (idx < T), v, np.nan)


def kernel_mean(Z, ok, x, sigma, dts, c, tau_h, exclude_self=True):
    """sum_{j,s} phi * z / sum phi with phi = exp(-|dx|/sigma - |dt - dx/c|/tau).

    dx = x_i - x_j (target minus source), dt = t - s in hours; s = t - k for k in dts.
    """
    Zf = np.nan_to_num(Z, nan=0.0).astype(np.float32); W = ok.astype(np.float32)
    dx = x[:, None] - x[None, :]                      # (i, j)
    num = np.zeros_like(Zf); den = np.zeros_like(Zf)
    T = Z.shape[0]
    for k in dts:
        dt = k * DT_H
        K = np.exp(-np.abs(dx) / sigma - np.abs(dt - dx / c) / tau_h).astype(np.float32)
        if exclude_self and k == 0:
            np.fill_diagonal(K, 0.0)
        # source rows s = t - k
        if k >= 0:
            src_z, src_w = Zf[: T - k], W[: T - k]
            num[k:] += src_z @ K.T; den[k:] += src_w @ K.T
        else:
            src_z, src_w = Zf[-k:], W[-k:]
            num[: T + k] += src_z @ K.T; den[: T + k] += src_w @ K.T
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 1e-6, num / den, np.nan).astype(np.float32), den


def grid_features(P):
    """Return dict of (T, n) float32 feature grids plus the input arrays."""
    V, Q, ok = inputs(P)
    T, n = V.shape
    lanes = P.lanes.astype(np.float32)[None, :]
    vf = P.vf.astype(np.float32)[None, :]
    ql = Q / lanes; vr = V / vf
    x = np.concatenate([[0.0], np.cumsum(P.length[:-1])]).astype(np.float64)
    F = {}
    p, nx = prev_next(V, ok)
    F["prev_vr"] = take(vr, p); F["next_vr"] = take(vr, nx)
    F["prev_ql"] = take(ql, p); F["next_ql"] = take(ql, nx)
    t = np.arange(T)[:, None]
    gp = np.where(p >= 0, t - p, 300).astype(np.float32); gn = np.where(nx < T, nx - t, 300).astype(np.float32)
    F["gap_prev"] = gp; F["gap_next"] = gn
    wp = np.where(np.isfinite(F["prev_vr"]), 1.0 / np.maximum(gp, 1), 0); wn = np.where(np.isfinite(F["next_vr"]), 1.0 / np.maximum(gn, 1), 0)
    with np.errstate(invalid="ignore", divide="ignore"):
        F["lin_vr"] = (np.nan_to_num(F["prev_vr"]) * wp + np.nan_to_num(F["next_vr"]) * wn) / (wp + wn)
        F["lin_ql"] = (np.nan_to_num(F["prev_ql"]) * wp + np.nan_to_num(F["next_ql"]) * wn) / (wp + wn)
    vfree = float(np.nanmedian(P.vf)); wcong = float(np.nanmedian(P.w))
    for tag, sig in (("s1", 1.0), ("s4", 4.0)):
        # same-time spatial, free-flow characteristic and congested characteristic
        F[f"sp_{tag}_ql"], d0 = kernel_mean(Q, ok, x, sig, [0], 1e9, 1.0); F[f"sp_{tag}_ql"] /= lanes
        F[f"sp_{tag}_vr"], _ = kernel_mean(vr, ok, x, sig, [0], 1e9, 1.0)
        F[f"ff_{tag}_ql"], _ = kernel_mean(Q, ok, x, sig, [-2, -1, 0, 1, 2], vfree, 2.5 / 60); F[f"ff_{tag}_ql"] /= lanes
        F[f"ff_{tag}_vr"], _ = kernel_mean(vr, ok, x, sig, [-2, -1, 0, 1, 2], vfree, 2.5 / 60)
        F[f"cg_{tag}_ql"], _ = kernel_mean(Q, ok, x, sig, [-3, -2, -1, 0, 1, 2, 3], -wcong, 2.5 / 60); F[f"cg_{tag}_ql"] /= lanes
        F[f"cg_{tag}_vr"], _ = kernel_mean(vr, ok, x, sig, [-3, -2, -1, 0, 1, 2, 3], -wcong, 2.5 / 60)
        F[f"den_{tag}"] = d0
    # nearest observed neighbours up/down stream at the same time (by link order)
    for side, rng in (("up", range(1, 6)), ("dn", range(1, 6))):
        best_v = np.full((T, n), np.nan, np.float32); best_q = best_v.copy(); dist = np.full((T, n), 99.0, np.float32)
        for k in rng:
            sh = -k if side == "up" else k
            vs = np.full((T, n), np.nan, np.float32); qs = vs.copy()
            if sh < 0:
                vs[:, -sh:] = vr[:, :sh]; qs[:, -sh:] = Q[:, :sh]
            else:
                vs[:, :-sh] = vr[:, sh:]; qs[:, :-sh] = Q[:, sh:]
            take_it = np.isnan(best_v) & np.isfinite(vs)
            best_v[take_it] = vs[take_it]; best_q[take_it] = qs[take_it]; dist[take_it] = k
        F[f"{side}_vr"] = best_v; F[f"{side}_ql"] = best_q / lanes; F[f"{side}_k"] = dist
    # local congestion indicator: min speed ratio in +-2 links, +-2 steps
    vr_f = np.where(np.isfinite(vr), vr, 9.0)
    mn = vr_f.copy()
    for a in (-2, -1, 1, 2):
        mn = np.minimum(mn, np.roll(vr_f, a, axis=0))
    mn2 = mn.copy()
    for b in (-2, -1, 1, 2):
        mn2 = np.minimum(mn2, np.roll(mn, b, axis=1))
    F["min_vr_nb"] = np.where(mn2 < 9, mn2, np.nan)
    return F, V, Q, ok


STATIC = ["vf", "capl", "lanes", "length", "pos", "w", "kcl"]


def rows(P, F, r, c):
    """Assemble a feature matrix for cells (r, c)."""
    ts = P.times[r]
    hh = np.array([int(s[11:13]) for s in ts]); mm = np.array([int(s[14:16]) for s in ts])
    tod = (hh + mm / 60.0).astype(np.float32)
    import pandas as pd
    dow = pd.to_datetime(pd.Series(ts).str[:10]).dt.dayofweek.to_numpy().astype(np.float32)
    x = np.concatenate([[0.0], np.cumsum(P.length[:-1])])
    st = {"vf": P.vf, "capl": P.cap / P.lanes, "lanes": P.lanes, "length": P.length,
          "pos": x / max(x[-1], 1), "w": P.w, "kcl": P.kc / P.lanes}
    cols = [tod, dow, P.regime[r].astype(np.float32)] + [st[k][c].astype(np.float32) for k in STATIC]
    names = ["tod", "dow", "regime"] + STATIC
    for k in sorted(F):
        cols.append(F[k][r, c]); names.append(k)
    return np.column_stack(cols).astype(np.float32), names
