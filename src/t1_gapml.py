"""Task 1 cells inside the Task 2 blanks: a dedicated model (residual on top of the per-link linear interpolation).

In validation/private the rows T+5..T+90 (local rows t+1..t+18) of every scored Task 2 window are blank on every link;
the Task 1 model never saw that in training. This module builds, for one window, per (gap row, link) features from
what a participant sees: the unmasked window history T-60..T-5, the T1-masked rows T and after the blank, the queue
geometry before/after the blank, the weekday x time-of-day demand profile, and static physics (no position / panel id).

view_train(P, t)   -> local (61, n) V, Q arrays of a TRAIN window as validation would show it
view_test(P, t)    -> same for validation/private (the cache is already the masked layer)
gap_features(...)  -> X (18*n, F) float32, names, base_v (km/h), base_q (vph/lane), rows k (1..18), links j
"""
from __future__ import annotations
import numpy as np

G, PRE, POST, L0 = 18, 12, 12, 24          # gap rows, pre rows (t-12..t), post rows (t+19..t+30), local row of t
R = L0 + G + POST + 7                       # local rows t-24 .. t+36
NB = (-5, -2, -1, 1, 2, 5)


def _local(A, t, T, fill=np.nan):
    out = np.full((R,) + A.shape[1:], fill, dtype=np.float64)
    a, b = t - L0, t - L0 + R
    lo, hi = max(a, 0), min(b, T)
    out[lo - a:hi - a] = A[lo:hi]
    return out


def view_train(P, t):
    V = _local(P.V, t, P.T); Q = _local(P.Q, t, P.T)
    bl = _local(P.blank, t, P.T, fill=0).astype(bool)
    bl[L0 - PRE:L0] = False                                  # window history T-60..T-5 is published unmasked
    V[bl] = np.nan; Q[bl] = np.nan
    V[L0 + 1:L0 + 1 + G] = np.nan; Q[L0 + 1:L0 + 1 + G] = np.nan
    ok = np.isfinite(V) & np.isfinite(Q); V[~ok] = np.nan; Q[~ok] = np.nan
    return V, Q


def view_test(P, t):
    V = _local(P.V, t, P.T); Q = _local(P.Q, t, P.T)
    V[L0 + 1:L0 + 1 + G] = np.nan; Q[L0 + 1:L0 + 1 + G] = np.nan   # already blank in the release; enforce
    ok = np.isfinite(V) & np.isfinite(Q); V[~ok] = np.nan; Q[~ok] = np.nan
    return V, Q


def _first(M, rows):
    """first finite value along the given local row order -> value, row (NaN/-1 when none)."""
    S = M[rows]; f = np.isfinite(S); has = f.any(axis=0); i = np.argmax(f, axis=0)
    cols = np.arange(M.shape[1])
    val = np.where(has, S[i, cols], np.nan); row = np.where(has, np.asarray(rows)[i], -1)
    return val, row


def _nanmean(M):
    with np.errstate(all="ignore"):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return np.nanmean(M, axis=0)


def _shift(a, d):
    out = np.full_like(a, np.nan)
    if d > 0:
        out[..., :-d] = a[..., d:]
    elif d < 0:
        out[..., -d:] = a[..., :d]
    else:
        out[...] = a
    return out


def _segments(q):
    idx = np.flatnonzero(q)
    if not len(idx):
        return []
    segs = [[idx[0], idx[0]]]
    for i in idx[1:]:
        if i - segs[-1][1] <= 2:
            segs[-1][1] = i
        else:
            segs.append([i, i])
    return segs


def _seg_rel(q, x):
    """nearest queue segment to every link: tail/head positions relative to the link (km), NaN if no queue."""
    n = len(x); sg = _segments(q)
    if not sg:
        return np.full(n, np.nan), np.full(n, np.nan)
    sx = np.array([[x[a], x[b]] for a, b in sg])
    d = np.where(x[:, None] < sx[None, :, 0], sx[None, :, 0] - x[:, None], np.where(x[:, None] > sx[None, :, 1], x[:, None] - sx[None, :, 1], 0.0))
    j = d.argmin(axis=1)
    return sx[j, 0] - x, sx[j, 1] - x


def _dq(q, x):
    n = len(x); qi = np.flatnonzero(q); ii = np.arange(n)
    if not len(qi):
        return np.full(n, 30.0), np.full(n, 30.0)
    pl = np.searchsorted(qi, ii, "right"); pr = np.searchsorted(qi, ii, "left")
    up = np.where(pl > 0, x - x[qi[np.maximum(pl - 1, 0)]], 30.0)
    dn = np.where(pr < len(qi), x[qi[np.minimum(pr, len(qi) - 1)]] - x, 30.0)
    return np.minimum(up, 30.0), np.minimum(dn, 30.0)


def _dow(day):
    return int(np.datetime64(day).astype("datetime64[D]").view("int64") % 7)


def gap_features(P, V, Q, t, cond, prof, dmin, x):
    """V, Q: local (R, n) views. prof: (7, 288, n) weekday x ToD mean flow. Returns X (G*n, F), names, base_v, base_q, k, j."""
    n = P.n; vf, cap, lanes = P.vf, P.cap, P.lanes
    vr, qc = V / vf[None, :], Q / cap[None, :]
    pre_rows = list(range(L0, L0 - PRE - 1, -1)); post_rows = list(range(L0 + G + 1, L0 + G + 1 + POST))
    pv, pa = _first(vr, pre_rows); pq, _ = _first(qc, pre_rows)
    nv, nb = _first(vr, post_rows); nq, _ = _first(qc, post_rows)
    pre_m_v, pre_m_q = _nanmean(vr[L0 - 3:L0 + 1]), _nanmean(qc[L0 - 3:L0 + 1])
    pre_o_v = _nanmean(vr[L0 - 9:L0 - 5])
    post_m_v, post_m_q = _nanmean(vr[L0 + G + 1:L0 + G + 5]), _nanmean(qc[L0 + G + 1:L0 + G + 5])
    post_o_v = _nanmean(vr[L0 + G + 7:L0 + G + 11])
    # queue states before / after the blank (km/h threshold v_cut)
    qpre = np.nan_to_num(pv * vf, nan=1e9) <= P.vcut; qpost = np.nan_to_num(nv * vf, nan=1e9) <= P.vcut
    tp, hp = _seg_rel(qpre, x); tn, hn = _seg_rel(qpost, x)
    dup0, ddn0 = _dq(qpre, x); dup1, ddn1 = _dq(qpost, x)
    # demand profile
    day = str(P.times[t])[:10]; pr = prof[_dow(day)]; sT = t % 288
    def lev(rows):
        acc = []
        for r in rows:
            s = (sT + (r - L0)) % 288
            acc.append(Q[r] / np.maximum(pr[s], 1.0))
        return _nanmean(np.stack(acc))
    lv0, lv1 = lev([L0 - 1, L0 - 2, L0 - 3]), lev([L0 + G + 1, L0 + G + 2, L0 + G + 3])
    st = dict(lanes=lanes, capl=cap / lanes / 2000.0, vf=vf / 110.0, w=P.w, dmin_rel=dmin / cap, length=P.length)
    win = dict(cond=1.0 if cond == "queue_ongoing" else 0.0, dow=float(_dow(day)), nq_pre=float(qpre.sum()), nq_post=float(qpost.sum()))
    cols = {}; names = None
    blocks = []
    for k in range(1, G + 1):
        i = L0 + k
        a = np.where(pa >= 0, pa, np.nan); b = np.where(nb >= 0, nb, np.nan)
        w = np.where(np.isfinite(a) & np.isfinite(b), (i - a) / (b - a), np.where(np.isfinite(a), 0.0, 1.0))
        vl = np.where(np.isfinite(pv) & np.isfinite(nv), pv * (1 - w) + nv * w, np.where(np.isfinite(pv), pv, nv))
        ql = np.where(np.isfinite(pq) & np.isfinite(nq), pq * (1 - w) + nq * w, np.where(np.isfinite(pq), pq, nq))
        tf = k / (G + 1.0)
        f = {"k": np.full(n, float(k)), "w": w, "pre_age": L0 - a, "post_age": b - (L0 + G + 1),
             "pre_v": pv, "pre_q": pq, "post_v": nv, "post_q": nq, "pre_m_v": pre_m_v, "pre_m_q": pre_m_q, "pre_o_v": pre_o_v,
             "post_m_v": post_m_v, "post_m_q": post_m_q, "post_o_v": post_o_v, "v_lin": vl, "q_lin": ql,
             "tail_pre": tp, "head_pre": hp, "tail_post": tn, "head_post": hn,
             "tail_int": tp + tf * (tn - tp), "head_int": hp + tf * (hn - hp),
             "dup_pre": dup0, "ddn_pre": ddn0, "dup_post": dup1, "ddn_post": ddn1}
        s_i = (sT + k) % 288
        sa = (sT + np.nan_to_num(a, nan=L0) - L0).astype(int) % 288; sb = (sT + np.nan_to_num(b, nan=L0 + G + 1) - L0).astype(int) % 288
        cidx = np.arange(n)
        plin = pr[sa, cidx] * (1 - w) + pr[sb, cidx] * w
        f["curv"] = pr[s_i] / np.maximum(plin, 1.0)
        f["dem_pre"] = lv0 * pr[s_i] / cap; f["dem_post"] = lv1 * pr[s_i] / cap
        f["dem_mix"] = (1 - tf) * f["dem_pre"] + tf * f["dem_post"]
        f["tod"] = np.full(n, ((t + k) % 288) / 12.0)
        for d in NB:
            f[f"nb{d}_vlin"] = _shift(vl, d); f[f"nb{d}_pre_v"] = _shift(pv, d); f[f"nb{d}_post_v"] = _shift(nv, d)
        for d in (-1, 1):
            f[f"nb{d}_qlin"] = _shift(ql, d)
        for kk, vv in st.items():
            f[kk] = vv
        for kk, vv in win.items():
            f[kk] = np.full(n, vv)
        if names is None:
            names = sorted(f)
        blocks.append(np.column_stack([f[c] for c in names]).astype(np.float32))
    X = np.vstack(blocks)
    iv, iq = names.index("v_lin"), names.index("q_lin")
    vfr = np.tile(vf, G); capr = np.tile(cap, G); lnr = np.tile(lanes, G)
    base_v = X[:, iv] * vfr; base_q = X[:, iq] * capr / lnr
    kk = np.repeat(np.arange(1, G + 1), n); jj = np.tile(np.arange(n), G)
    return X, names, base_v.astype(np.float64), base_q.astype(np.float64), kk, jj


PARAMS = dict(objective="regression", learning_rate=0.05, num_leaves=127, min_data_in_leaf=200, feature_fraction=0.8,
              bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, num_threads=8, verbose=-1, seed=3, force_row_wise=True,
              deterministic=True)
