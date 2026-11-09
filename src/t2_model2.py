"""Task 2 v2: v1 features + queue-segment kinematics (tail/head positions, empirical and
flow-based shock speeds, step-specific predicted tail offset)."""
from __future__ import annotations
import os, pickle, sys
from pathlib import Path
import numpy as np, lightgbm as lgb
from cachelib import load, T2_PANELS
from t2_windows import H, F, score
from t2_model import PanelInputs, link_features, priors, best_set, FOLDS, PARAMS, OUT


def segments(q):
    """Contiguous runs of queued links (allowing one-link gaps)."""
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


SEMP = os.environ.get("TFB_T2_SEMP", "")      # 10/22 review B: "seg" = per-segment s_emp, clipped to +-25 km/h ("" = as before)


def semp_seg(hist, x, cut, sg_now):
    """s_emp per current segment (10/22 review B): walk back from the newest history row and follow the segment (the row's
    segment that overlaps the followed one most, gaps up to 2 km allowed), fit tail / head position against time; slopes
    clipped to +-25 km/h (backward-wave physics 23-28 km/h; the global version reached 197 km/h when the largest segment
    switched). Returns (tail slopes, head slopes) per segment of sg_now (NaN with < 3 rows)."""
    rows = [[(x[a], x[b]) for a, b in segments(np.nan_to_num(hist[k], nan=1e9) <= cut)] for k in range(hist.shape[0])]
    st, sh = [], []
    for a, b in sg_now:
        ta, tb = x[a], x[b]; rk, tt, th = [], [], []
        for k in range(len(rows) - 1, -1, -1):
            if not rows[k]:
                continue
            ov = [min(tb, cb) - max(ta, ca) for ca, cb in rows[k]]
            i = int(np.argmax(ov))
            if ov[i] < -2.0:
                continue
            ta, tb = rows[k][i]; rk.append(k); tt.append(ta); th.append(tb)
        if len(rk) >= 3:
            r = np.array(rk) * (5 / 60.0)
            st.append(float(np.clip(np.polyfit(r, tt, 1)[0], -25, 25))); sh.append(float(np.clip(np.polyfit(r, th, 1)[0], -25, 25)))
        else:
            st.append(np.nan); sh.append(np.nan)
    return np.array(st), np.array(sh)


def kin_features(PI, t):
    P = PI.P; x = PI.x
    V, Q, tl = PI.window(t)
    cut = P.vcut[None, :]
    hist = V[max(tl - H, 0):tl + 1]
    # tail/head per history row for the largest segment at each row
    tails, heads, rows_ = [], [], []
    for k in range(hist.shape[0]):
        qk = np.nan_to_num(hist[k], nan=1e9) <= cut[0]
        sg = segments(qk)
        if sg:
            a, b = max(sg, key=lambda s: s[1] - s[0])
            tails.append(x[a]); heads.append(x[b]); rows_.append(k)
    fin = np.isfinite(hist)
    idx = np.where(fin, np.arange(hist.shape[0])[:, None], -1).max(axis=0)
    cols = np.arange(P.n)
    last_v = np.where(idx >= 0, hist[np.maximum(idx, 0), cols], np.nan)
    Qh = Q[max(tl - H, 0):tl + 1]
    last_q = np.where(idx >= 0, Qh[np.maximum(idx, 0), cols], np.nan)
    lq = np.nan_to_num(last_v, nan=1e9) <= P.vcut
    sg = segments(lq)
    n = P.n
    out = {k: np.full(n, np.nan) for k in ("seg_tail", "seg_head", "seg_len", "s_flow", "s_emp_tail", "s_emp_head")}
    if sg:
        # assign each link to the nearest segment (by distance to [tail, head])
        segx = np.array([[x[a], x[b]] for a, b in sg])
        d = np.where(x[:, None] < segx[None, :, 0], segx[None, :, 0] - x[:, None],
                     np.where(x[:, None] > segx[None, :, 1], x[:, None] - segx[None, :, 1], 0.0))
        j = d.argmin(axis=1)
        sflow = []
        for a, b in sg:
            up = [i for i in range(max(a - 4, 0), a) if np.isfinite(last_q[i]) and not lq[i]]
            qin = np.mean(last_q[up]) if up else np.nan
            kin = qin / np.mean(P.vf[up]) if up else np.nan
            inside = [i for i in range(a, b + 1) if np.isfinite(last_q[i]) and np.isfinite(last_v[i])]
            qq = np.mean(last_q[inside]) if inside else np.nan
            kq = np.mean(last_q[inside] / np.maximum(last_v[inside], 1)) if inside else np.nan
            s = (qin - qq) / (kin - kq) if np.isfinite(qin) and np.isfinite(qq) and abs(kin - kq) > 1e-6 else np.nan
            sflow.append(np.clip(s, -60, 60) if np.isfinite(s) else np.nan)
        out["seg_tail"] = segx[j, 0] - x; out["seg_head"] = segx[j, 1] - x
        out["seg_len"] = segx[j, 1] - segx[j, 0]; out["s_flow"] = np.array(sflow)[j]
    if len(rows_) >= 3:
        r = np.array(rows_) * (5 / 60.0)
        out["s_emp_tail"][:] = np.polyfit(r, tails, 1)[0]; out["s_emp_head"][:] = np.polyfit(r, heads, 1)[0]
    if SEMP == "seg":                                   # per segment where there is a current one, else the global fit, clipped
        out["s_emp_tail"] = np.clip(out["s_emp_tail"], -25, 25); out["s_emp_head"] = np.clip(out["s_emp_head"], -25, 25)
        if sg:
            st, sh = semp_seg(hist, x, cut[0], sg)
            out["s_emp_tail"] = np.where(np.isfinite(st[j]), st[j], out["s_emp_tail"])
            out["s_emp_head"] = np.where(np.isfinite(sh[j]), sh[j], out["s_emp_head"])
    names = sorted(out)
    return np.column_stack([out[k] for k in names]), names


def feats(PI, t, cond, pr):
    X1, n1 = link_features(PI, t, cond, *pr)
    X2, n2 = kin_features(PI, t)
    return np.column_stack([X1, X2]).astype(np.float32), n1 + n2


def step_block(X, names, s):
    """Append step and step-specific predicted tail offsets."""
    i_tail = names.index("seg_tail"); i_sf = names.index("s_flow"); i_se = names.index("s_emp_tail")
    h = s * 5 / 60.0
    pt_flow = X[:, i_tail] + np.nan_to_num(X[:, i_sf]) * h
    pt_emp = X[:, i_tail] + np.nan_to_num(X[:, i_se]) * h
    return np.column_stack([X, np.full(len(X), s), pt_flow, pt_emp]).astype(np.float32)


def design(W, PIs, pr, cond):
    X, y, g = [], [], []
    for w in W:
        if w["cond"] != cond:
            continue
        Xl, names = feats(PIs[w["panel"]], w["t"], cond, pr[w["panel"]])
        if cond == "queue_onset":
            X.append(Xl); y.append(w["label"][-1]); g.append(w["elig"][-1])
        else:
            for s in range(F):
                X.append(step_block(Xl, names, s + 1)); y.append(w["label"][s]); g.append(w["elig"][s])
    return np.vstack(X), np.concatenate(y).astype(np.float32), np.concatenate(g)


def predict_window(m, PI, t, cond, pr):
    Xl, names = feats(PI, t, cond, pr)
    n = len(Xl)
    if cond == "queue_onset":
        p = np.zeros((F, n)); p[-1] = m.predict(Xl, num_threads=16)
    else:
        p = np.stack([m.predict(step_block(Xl, names, s + 1), num_threads=16) for s in range(F)])
    return p


if __name__ == "__main__":
    W = pickle.load(open(r"C:/Users/user/tfb/work/t2_windows_train.pkl", "rb"))
    Ps = {p: load(p, "train") for p in T2_PANELS}
    PIs = {p: PanelInputs(Ps[p]) for p in T2_PANELS}
    nmap = {p: Ps[p].n for p in T2_PANELS}
    oof = [None] * len(W)
    for fi, months in enumerate(FOLDS):
        pr = priors(W, nmap, set(months))
        tr = [w for w in W if w["date"][:7] not in months]
        for cond in ("queue_onset", "queue_ongoing"):
            X, y, g = design(tr, PIs, pr, cond)
            m = lgb.train(PARAMS, lgb.Dataset(X[g], y[g]), 600 if cond == "queue_onset" else 800)
            for k, w in enumerate(W):
                if w["cond"] == cond and w["date"][:7] in months:
                    oof[k] = predict_window(m, PIs[w["panel"]], w["t"], cond, pr[w["panel"]])
        print("fold", fi, flush=True)
    pickle.dump(oof, open(OUT / "oof_v2.pkl", "wb"))
    preds = [best_set(o.ravel(), w["elig"].ravel()).reshape(o.shape) for o, w in zip(oof, W)]
    print("v2 expected-IoU set:", np.round(score(W, preds), 4))
