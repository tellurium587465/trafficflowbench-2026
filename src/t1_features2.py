"""Task 1 features v2.

Changes from v1:
- inputs keep every released finite value (ineligible cells carry the same noise level)
- per-link static offsets learned on train months Jun-Jan: median free-flow v/vf and
  median flow ratio to the same-time spatial kernel
- same-time corridor-wide mean of v/vf and q/cap (common factor), leave-one-out
- extra kernels: directional same-time kernels on total flow, short-range characteristic kernel,
  temporal-only kernel on the same link
"""
from __future__ import annotations
from functools import lru_cache
import numpy as np, pandas as pd
from cachelib import load
from t1_features import prev_next, take, kernel_mean, DT_H

STAT_MONTHS = ("2030-06", "2030-07", "2030-08", "2030-09", "2030-10", "2030-11", "2030-12", "2031-01")


def inputs2(P):
    V = P.V.astype(np.float32).copy(); Q = P.Q.astype(np.float32).copy()
    if P.split == "train":
        V[P.blank] = np.nan; Q[P.blank] = np.nan
    ok = np.isfinite(V) & np.isfinite(Q)
    V[~ok] = np.nan; Q[~ok] = np.nan
    return V, Q, ok


import os
ADAPT = os.environ.get("TFB_T1_ADAPT", "0") == "1"
NOMEM = os.environ.get("TFB_T1_NOMEM", "0") == "1"    # 10/10: drop location/time memory from the model rows (panel id is zeroed by the caller)
NOMEM_DROP = {"tod", "dow", "vf", "capl", "lanes", "length", "pos", "w", "kcl", "_vmed", "_qratio"}   # val/private: link offsets from the split's own released cells


@lru_cache(maxsize=None)
def link_statics(panel, split="train"):
    # validation/private months redraw per-link free-flow speed (~1%, train months agree to 0.03%)
    own = ADAPT and split != "train"
    P = load(panel, split if own else "train")
    rowsel = np.ones(P.T, bool) if own else np.array([s[:7] in STAT_MONTHS for s in P.times])
    ok = np.isfinite(P.V) & P.E & (P.V > 0.9 * P.vf[None, :]) & rowsel[:, None]
    vr = np.where(ok, P.V / P.vf[None, :], np.nan)
    vmed = np.nanmedian(vr[::3], axis=0)
    x = np.concatenate([[0.0], np.cumsum(P.length[:-1])])
    sub = slice(0, None, 7)
    Q = np.where(ok, P.Q, np.nan)[sub]
    km, _ = kernel_mean(Q, ok[sub], x, 1.0, [0], 1e9, 1.0)
    qratio = np.nanmedian(Q / km, axis=0)
    return np.nan_to_num(vmed, nan=1.0).astype(np.float32), np.nan_to_num(qratio, nan=1.0).astype(np.float32)


def shifted(M, k):
    out = np.full_like(M, np.nan)
    if k > 0:
        out[k:] = M[:-k]
    elif k < 0:
        out[:k] = M[-k:]
    else:
        out[:] = M
    return out


def grid_features2(P):
    V, Q, ok = inputs2(P)
    T, n = V.shape
    lanes = P.lanes.astype(np.float32)[None, :]
    vf = P.vf.astype(np.float32)[None, :]
    vmed, qratio = link_statics(P.panel, P.split)
    vr = V / (vf * vmed[None, :]); ql = Q / lanes
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
    # temporal-only kernel (same link, +-1..3 steps, weight 1/|dt|)
    num = np.zeros((T, n), np.float32); den = np.zeros((T, n), np.float32)
    for k, wt in ((1, 1.0), (2, 0.5), (3, 0.33)):
        for s in (k, -k):
            z = shifted(ql, s); m = np.isfinite(z)
            num += np.where(m, z, 0) * wt; den += m * wt
    with np.errstate(invalid="ignore", divide="ignore"):
        F["tk_ql"] = np.where(den > 0, num / den, np.nan)
    vfree = float(np.nanmedian(P.vf)); wcong = float(np.nanmedian(P.w))
    Qadj = Q / qratio[None, :]
    for tag, sig in (("s05", 0.5), ("s1", 1.0), ("s3", 3.0)):
        F[f"sp_{tag}_ql"], d0 = kernel_mean(Qadj, ok, x, sig, [0], 1e9, 1.0)
        F[f"sp_{tag}_ql"] *= qratio[None, :] / lanes
        F[f"sp_{tag}_vr"], _ = kernel_mean(vr, ok, x, sig, [0], 1e9, 1.0)
        F[f"den_{tag}"] = d0
    for tag, sig in (("s05", 0.5), ("s2", 2.0)):
        F[f"ff_{tag}_ql"], _ = kernel_mean(Qadj, ok, x, sig, [-2, -1, 0, 1, 2], vfree, 2.5 / 60)
        F[f"ff_{tag}_ql"] *= qratio[None, :] / lanes
        F[f"ff_{tag}_vr"], _ = kernel_mean(vr, ok, x, sig, [-2, -1, 0, 1, 2], vfree, 2.5 / 60)
        F[f"cg_{tag}_ql"], _ = kernel_mean(Qadj, ok, x, sig, [-3, -2, -1, 0, 1, 2, 3], -wcong, 2.5 / 60)
        F[f"cg_{tag}_ql"] *= qratio[None, :] / lanes
        F[f"cg_{tag}_vr"], _ = kernel_mean(vr, ok, x, sig, [-3, -2, -1, 0, 1, 2, 3], -wcong, 2.5 / 60)
    # common factor: corridor mean at the same time, leave-one-out, free-flow-ish cells only
    fr = ok & (vr > 0.9)
    s_v = np.nansum(np.where(fr, vr, 0), axis=1, keepdims=True); c_v = fr.sum(axis=1, keepdims=True)
    F["cm_vr"] = np.where(c_v - fr > 0, (s_v - np.where(fr, vr, 0)) / np.maximum(c_v - fr, 1), np.nan)
    qc = Q / P.cap[None, :]
    s_q = np.nansum(np.where(ok, qc, 0), axis=1, keepdims=True); c_q = ok.sum(axis=1, keepdims=True)
    F["cm_qc"] = np.where(c_q - ok > 0, (s_q - np.where(ok, qc, 0)) / np.maximum(c_q - ok, 1), np.nan)
    # nearest observed up/down (by link order), total flow converted to own lanes
    for side in ("up", "dn"):
        best_v = np.full((T, n), np.nan, np.float32); best_q = best_v.copy(); dist = np.full((T, n), 99.0, np.float32)
        for k in range(1, 6):
            sh = -k if side == "up" else k
            vs = np.full((T, n), np.nan, np.float32); qs = vs.copy()
            if sh < 0:
                vs[:, -sh:] = vr[:, :sh]; qs[:, -sh:] = Qadj[:, :sh]
            else:
                vs[:, :-sh] = vr[:, sh:]; qs[:, :-sh] = Qadj[:, sh:]
            tk = np.isnan(best_v) & np.isfinite(vs)
            best_v[tk] = vs[tk]; best_q[tk] = qs[tk]; dist[tk] = k
        F[f"{side}_vr"] = best_v; F[f"{side}_ql"] = best_q * qratio[None, :] / lanes; F[f"{side}_k"] = dist
    vr_f = np.where(np.isfinite(vr), vr, 9.0)
    mn = vr_f.copy()
    for a in (-2, -1, 1, 2):
        mn = np.minimum(mn, np.roll(vr_f, a, axis=0))
    mn2 = mn.copy()
    for b in (-2, -1, 1, 2):
        mn2 = np.minimum(mn2, np.roll(mn, b, axis=1))
    F["min_vr_nb"] = np.where(mn2 < 9, mn2, np.nan)
    F["_vmed"] = np.broadcast_to(vmed[None, :], (T, n)); F["_qratio"] = np.broadcast_to(qratio[None, :], (T, n))
    return F, V, Q, ok


STATIC = ["vf", "capl", "lanes", "length", "pos", "w", "kcl"]


def rows2(P, F, r, c):
    ts = P.times[r]
    hh = np.array([int(s[11:13]) for s in ts]); mm = np.array([int(s[14:16]) for s in ts])
    tod = (hh + mm / 60.0).astype(np.float32)
    dow = pd.to_datetime(pd.Series(ts).str[:10]).dt.dayofweek.to_numpy().astype(np.float32)
    x = np.concatenate([[0.0], np.cumsum(P.length[:-1])])
    st = {"vf": P.vf, "capl": P.cap / P.lanes, "lanes": P.lanes, "length": P.length,
          "pos": x / max(x[-1], 1), "w": P.w, "kcl": P.kc / P.lanes}
    cols = [tod, dow, P.regime[r].astype(np.float32)] + [st[k][c].astype(np.float32) for k in STATIC]
    names = ["tod", "dow", "regime"] + STATIC
    for k in sorted(F):
        cols.append(F[k][r, c]); names.append(k)
    if ADAPT and P.split != "train" and os.environ.get("TFB_T1_FEAT_TRAINSTAT", "0") == "1":
        # audit F3: keep the adapted statics for normalisation/base, but feed the model the train values it was
        # fitted on (adapted values are unseen link-level numbers and can act as a shifted link identifier)
        tv, tq = link_statics(P.panel, "train")
        cols[names.index("_vmed")] = tv[c]; cols[names.index("_qratio")] = tq[c]
    if NOMEM:                                                   # CAUSAL_MAP P1/P7: no time, link statics or link-level offsets
        keep = [i for i, n in enumerate(names) if n not in NOMEM_DROP]
        cols = [cols[i] for i in keep]; names = [names[i] for i in keep]
    return np.column_stack(cols).astype(np.float32), names


def base_of2(F):
    """Base estimates in physical units of (v/vf, q/lane)."""
    bv = np.where(np.isfinite(F["lin_vr"]), F["lin_vr"], 1.0) * F["_vmed"]
    bq = np.where(np.isfinite(F["ff_s05_ql"]), F["ff_s05_ql"], F["lin_ql"])
    return bv, bq
