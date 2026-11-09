"""Task 2 v1: per-link LightGBM with causal features (rows <= T only), month-grouped CV,
expected-IoU set selection per window."""
from __future__ import annotations
import pickle, sys, json
from pathlib import Path
import numpy as np, pandas as pd, lightgbm as lgb
from cachelib import load, T2_PANELS
from common import REL, WORK
from t2_windows import H, F, score, queue_matrix
from t1_features import inputs

OUT = Path(WORK) / "t2"; OUT.mkdir(parents=True, exist_ok=True)
NB_OFF = (-5, -3, -2, -1, 1, 2, 3, 5)
import os
NO_T = os.environ.get("TFB_T2_NOT", "0") == "1"
MASKHIST = os.environ.get("TFB_T2_MASKHIST", "0") == "1"     # 10/23: train windows with M3-type (masked) last-12-row history


class PanelInputs:
    """Causal input arrays for one panel/split, as a participant sees them."""

    def __init__(self, P):
        self.P = P
        # audit F4: ineligible cells (pct<75) carry the same noise -> TFB_T2_ALLFIN=1 keeps every finite value
        good = np.isfinite(P.V) & np.isfinite(P.Q) if os.environ.get("TFB_T2_ALLFIN", "0") == "1" else (P.P >= 75) & P.E
        if P.split == "train":
            self.full_v = np.where(good, P.V, np.nan); self.full_q = np.where(good, P.Q, np.nan)
            Vm, Qm, ok = inputs(P)
            self.mask_v, self.mask_q = Vm, Qm
        else:
            self.full_v = self.mask_v = np.where(good, P.V, np.nan)
            self.full_q = self.mask_q = np.where(good, P.Q, np.nan)
        # 10/22 review B (g59): q_today on the test side with the training filter (finite & pct >= 75, as t1_features.inputs);
        # off by default; M3 v2 +0.0003 (+1.2 SE, ongoing 26 windows) -> "undetermined", final A only
        self.qt_v = (np.where(np.isfinite(P.V) & np.isfinite(P.Q) & (P.P >= 75), P.V, np.nan)
                     if P.split != "train" and os.environ.get("TFB_T2_QTODAY_P75", "0") == "1" else None)
        x = np.concatenate([[0.0], np.cumsum(P.length[:-1])]); self.x = x
        self.pos = x / max(x[-1], 1)

    def window(self, t):
        """Rows from the start of the day through t (inclusive); history rows unmasked as released."""
        P = self.P
        d0 = t - (t % 288)
        V = self.mask_v[d0:t + 1].copy(); Q = self.mask_q[d0:t + 1].copy()
        a = max(t - H, d0)
        V[a - d0:t - d0] = self.full_v[a:t]; Q[a - d0:t - d0] = self.full_q[a:t]
        if MASKHIST and P.split == "train":   # 10/23 instrument v3 §2.1: M3-type history (the day's Task 1 mask also on T-12..T-1)
            V[a - d0:t - d0] = np.where(P.blank[a:t], np.nan, self.full_v[a:t]); Q[a - d0:t - d0] = np.where(P.blank[a:t], np.nan, self.full_q[a:t])
        if NO_T:   # conservative reading of the Task 2 rule: nothing at the origin row T itself
            V[t - d0] = np.nan; Q[t - d0] = np.nan
        return V, Q, t - d0


def link_features(PI, t, cond, prior_on, prior_og):
    P = PI.P
    V, Q, tl = PI.window(t)
    vr = V / P.vf[None, :]; qc = Q / P.cap[None, :]
    cut = (P.vcut / P.vf)[None, :]
    hv, hq = vr[max(tl - H, 0):tl + 1], qc[max(tl - H, 0):tl + 1]     # 13 rows incl. T
    fin = np.isfinite(hv)
    idx = np.where(fin, np.arange(hv.shape[0])[:, None], -1).max(axis=0)
    cols = np.arange(P.n)
    last_v = np.where(idx >= 0, hv[np.maximum(idx, 0), cols], np.nan)
    last_q = np.where(idx >= 0, hq[np.maximum(idx, 0), cols], np.nan)
    age = np.where(idx >= 0, hv.shape[0] - 1 - idx, 20).astype(float)
    qh = np.where(fin, hv <= cut, False)
    last_queue = (last_v <= cut[0]).astype(float); last_queue[np.isnan(last_v)] = np.nan
    with np.errstate(all="ignore"):
        f = {
            "pos": PI.pos, "vf": P.vf, "capl": P.cap / P.lanes, "lanes": P.lanes, "length": P.length, "w": P.w,
            "prior_on": prior_on, "prior_og": prior_og,
            "last_v": last_v, "last_q": last_q, "age": age, "last_queue": last_queue,
            "v_T": hv[-1], "v_T1": hv[-2], "v_T2": hv[-3], "q_T": hq[-1], "q_T1": hq[-2], "q_T2": hq[-3],
            "v_m3": np.nanmean(hv[-4:], axis=0), "v_m12": np.nanmean(hv, axis=0), "v_min": np.nanmin(np.where(fin, hv, np.inf), axis=0),
            "q_m3": np.nanmean(hq[-4:], axis=0), "q_m12": np.nanmean(hq, axis=0), "q_max": np.nanmax(np.where(np.isfinite(hq), hq, -np.inf), axis=0),
            "v_trend": np.nanmean(hv[-4:], axis=0) - np.nanmean(hv[:4], axis=0),
            "q_trend": np.nanmean(hq[-4:], axis=0) - np.nanmean(hq[:4], axis=0),
            "nq_hist": qh.sum(axis=0).astype(float),
            "q_today": np.where(np.isfinite(vr[:max(tl - H, 0)]), vr[:max(tl - H, 0)] <= cut, False).sum(axis=0).astype(float) if tl > H else np.zeros(P.n),
        }
        if getattr(PI, "qt_v", None) is not None and tl > H:   # TFB_T2_QTODAY_P75: same rows (day start .. T-H-1), training filter
            qv = PI.qt_v[t - tl:t - H] / P.vf[None, :]
            f["q_today"] = np.where(np.isfinite(qv), qv <= cut, False).sum(axis=0).astype(float)
    f["v_min"] = np.where(np.isfinite(f["v_min"]), f["v_min"], np.nan)
    f["q_max"] = np.where(np.isfinite(f["q_max"]), f["q_max"], np.nan)
    lq = np.nan_to_num(last_queue) > 0.5
    qi = np.flatnonzero(lq)
    ii = np.arange(P.n)
    if len(qi):
        pl = np.searchsorted(qi, ii, "left"); pr_ = np.searchsorted(qi, ii, "right")
        up = np.where(pl > 0, qi[np.maximum(pl - 1, 0)], -999)
        dn = np.where(pr_ < len(qi), qi[np.minimum(pr_, len(qi) - 1)], 9999)
        f["dq_up"] = np.where(up >= 0, PI.x - PI.x[np.clip(up, 0, P.n - 1)], 99.0)
        f["dq_dn"] = np.where(dn < P.n, PI.x[np.clip(dn, 0, P.n - 1)] - PI.x, 99.0)
        f["head_rel"] = PI.x[qi.max()] - PI.x; f["tail_rel"] = PI.x - PI.x[qi.min()]
    else:
        for k in ("dq_up", "dq_dn", "head_rel", "tail_rel"):
            f[k] = np.full(P.n, 99.0)
    f["n_queued_now"] = np.full(P.n, float(lq.sum())); f["n_queued_hist"] = np.full(P.n, float(qh.any(axis=0).sum()))
    for o in NB_OFF:
        j = np.clip(ii + o, 0, P.n - 1); valid = (ii + o >= 0) & (ii + o < P.n)
        f[f"nb{o}_v"] = np.where(valid, last_v[j], np.nan); f[f"nb{o}_q"] = np.where(valid, last_q[j], np.nan)
        f[f"nb{o}_qmax"] = np.where(valid, f["q_max"][j], np.nan)
    ts = P.times[t]; tod = int(ts[11:13]) + int(ts[14:16]) / 60
    f["tod"] = np.full(P.n, tod); f["dow"] = np.full(P.n, float(pd.Timestamp(ts[:10]).dayofweek))
    f["cond"] = np.full(P.n, 1.0 if cond == "queue_ongoing" else 0.0)
    names = sorted(f)
    return np.column_stack([f[k] for k in names]).astype(np.float32), names


def priors(W, panels_n, exclude_months):
    pr = {}
    for p, n in panels_n.items():
        on = np.zeros(n); og = np.zeros(n); non = nog = 0
        for w in W:
            if w["panel"] != p or w["date"][:7] in exclude_months:
                continue
            if w["cond"] == "queue_onset":
                on += w["label"][-1]; non += 1
            else:
                og += w["label"].any(axis=0); nog += 1
        pr[p] = (on / max(non, 1), og / max(nog, 1))
    return pr


FOLDS = [("2030-06", "2030-07", "2030-08"), ("2030-09", "2030-10", "2030-11"), ("2030-12", "2031-01", "2031-02")]
PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=63, min_data_in_leaf=50, feature_fraction=0.8,
              bagging_fraction=0.8, bagging_freq=1, lambda_l2=5.0, num_threads=16, verbose=-1, seed=7,
              force_row_wise=True)


def design(W, PIs, pr, cond):
    X, y, g, wid = [], [], [], []
    for k, w in enumerate(W):
        if w["cond"] != cond:
            continue
        Xl, names = link_features(PIs[w["panel"]], w["t"], cond, *pr[w["panel"]])
        if cond == "queue_onset":
            X.append(Xl); y.append(w["label"][-1]); g.append(w["elig"][-1]); wid.append(np.full(len(Xl), k))
        else:
            for s in range(F):
                X.append(np.column_stack([Xl, np.full(len(Xl), s + 1)])); y.append(w["label"][s]); g.append(w["elig"][s]); wid.append(np.full(len(Xl), k))
            names = names + ["step"]
    return np.vstack(X), np.concatenate(y).astype(np.float32), np.concatenate(g), np.concatenate(wid), names


def best_set(p, elig):
    """Choose the cell set maximizing E[I]/E[U] approx: sort by p, take top k."""
    pe = np.where(elig, p, 0.0)
    order = np.argsort(-pe)
    cs = np.cumsum(pe[order]); tot = pe.sum()
    k = np.arange(1, len(pe) + 1)
    eiou = cs / (k + tot - cs)
    kb = int(np.argmax(eiou)) + 1
    sel = np.zeros(len(p), bool); sel[order[:kb]] = True
    return sel


if __name__ == "__main__":
    W = pickle.load(open(r"C:/Users/user/tfb/work/t2_windows_train.pkl", "rb"))
    Ps = {p: load(p, "train") for p in T2_PANELS}
    PIs = {p: PanelInputs(Ps[p]) for p in T2_PANELS}
    nmap = {p: Ps[p].n for p in T2_PANELS}
    oof = [None] * len(W)
    for fi, months in enumerate(FOLDS):
        pr = priors(W, nmap, set(months))
        tr = [w for w in W if w["date"][:7] not in months]
        te_idx = [k for k, w in enumerate(W) if w["date"][:7] in months]
        for cond in ("queue_onset", "queue_ongoing"):
            X, y, g, _, names = design(tr, PIs, pr, cond)
            m = lgb.train(PARAMS, lgb.Dataset(X[g], y[g]), 600 if cond == "queue_onset" else 800)
            for k in te_idx:
                w = W[k]
                if w["cond"] != cond:
                    continue
                Xl, _ = link_features(PIs[w["panel"]], w["t"], cond, *pr[w["panel"]])
                n = len(Xl)
                if cond == "queue_onset":
                    p = m.predict(Xl, num_threads=16)
                    P6 = np.zeros((F, n)); P6[-1] = p
                else:
                    P6 = np.stack([m.predict(np.column_stack([Xl, np.full(n, s + 1)]), num_threads=16) for s in range(F)])
                oof[k] = P6
        print("fold", fi, "done", flush=True)
    pickle.dump(oof, open(OUT / "oof_v1.pkl", "wb"))
    for thr in (0.3, 0.4, 0.5):
        preds = [o >= thr for o in oof]
        print("thr", thr, "S_queue(all, onset, ongoing):", np.round(score(W, preds), 4))
    preds = [best_set(o.ravel(), w["elig"].ravel()).reshape(o.shape) for o, w in zip(oof, W)]
    print("expected-IoU set:", np.round(score(W, preds), 4))
