"""M3 v2 (10/22 evening; review D H1-H4, M1-M2, M5; agreed with the strategy session, METAGAME_REPLY §11). v1 stays as is
(t2_m3_gate.py / m3_stats.py) for the records; v2 changes the instrument, not the windows:
  1. post-stratified weights: within each condition, panels equal and weekend x origin-hour bins (<9, 9-12, 12-15, >=15)
     raked to the official windows of the target split ("private" = April, the main target; "validation" = March, to
     read the public LB); the effective sample size is reported.
  2. k as an expectation: a window's score is the mean over k of its score with k earlier days, k from the official
     distribution {0:.19, 1:.20, 2:.20, 3:.20, 4:.07, 5:.07, 6:.07}, truncated at the window's day (min(k, day)).
  3. selector-simulated blanks (replaces trap 4): for a window on day D with k days, the organiser selector replica
     (t2_selector.scan: chronological, 5 onset + 5 ongoing, 6 h spacing, coverage >= 0.5 on the masked month) is run
     from the start of day D-k up to the window's origin; the rows t+1..t+18 of every window it picks are hidden from
     the evidence (all links) and from X1's fine-tuning windows.
  4. fast summary: weighted per-condition differences, cluster bootstrap by matrix products, P(B>A) over 80-window
     redraws stratified 5 per panel x condition, a sign-flip permutation p-value over clusters, the number of windows
     that changed, and an optional seed-noise variance added to the SE.
usage: import m3v2"""
import functools, numpy as np, pandas as pd
from cachelib import load, T2_PANELS
from common import REL
from t2_windows import queue_matrix
import t2_selector

KP = np.array([0.19, 0.20, 0.20, 0.20, 0.07, 0.07, 0.07])
HBINS = [0, 9, 12, 15, 24]
MARCH_FIRST_DOW = 5                                             # 2031-03-01 is a Saturday


def is_holdout(day):
    """strategy §12 Q2 (10/22): M3 windows on calendar days = 0 mod 3 (0-based `day` + 1: the 3rd, 6th, ... 30th) are held
    out until the final A/B selection (11/4-5); constants fitted from now on (X1 gate, LightGBM weight, feature fixes,
    calibration) are judged on the other windows only."""
    return (np.asarray(day) + 1) % 3 == 0


def kdist(day, cap=None):
    """{k: prob} for a window on 0-based March day `day` (k earlier days available), optionally capped (cap = 3 for the
    evidence rule, which never looks further back): k = min(k_draw, day, cap)"""
    out = {}
    for k, p in enumerate(KP):
        kk = min(k, day, cap if cap is not None else 99); out[kk] = out.get(kk, 0.0) + p
    return out


@functools.lru_cache(maxsize=None)
def _official_bins(split):
    rows = []
    for p in T2_PANELS:
        for r in pd.read_csv(REL / "task2" / p / split / "window_index.csv").itertuples():
            rows.append(dict(cond=r.condition, we=pd.Timestamp(r.date).dayofweek >= 5, hb=int(np.digitize(int(r.forecast_origin[11:13]), HBINS[1:-1]))))
    D = pd.DataFrame(rows)
    return {c: g.groupby(["we", "hb"]).size().div(len(g)).to_dict() for c, g in D.groupby("cond")}


def weights(D, target="private", iters=50, extra=None):
    """D: cond, panel, day, hour (origin hour, 0-23) -> weights summing to 1 per condition (panels equal, weekend x hour
    bins raked to the target split's official windows). Returns (w, ess per condition).
    extra (10/23 instrument v3, the 'sub' weights): {cond: [(column of D, {value: target share}), ...]} raked as further
    margins, e.g. onset 'rare' -> {True: 0.40} (M4b) and ongoing persistence / T-1 queue-size bins (June official, G78)."""
    w = np.ones(len(D)); ess = {}; extra = extra or {}
    we = (((MARCH_FIRST_DOW + D["day"].to_numpy()) % 7) >= 5); hb = np.digitize(D["hour"].to_numpy(), HBINS[1:-1])
    tgt = _official_bins(target)
    for c in D.cond.unique():
        m = (D.cond == c).to_numpy(); pans = D.loc[m, "panel"].to_numpy(); cells = list(zip(we[m], hb[m])); wc = np.ones(m.sum())
        T = {cell: tgt[c].get(cell, 0.0) for cell in set(cells)}; s = sum(T.values()); T = {k: v / s for k, v in T.items()}   # target over cells present
        cl = np.array([hash(x) for x in cells]); keys = {hash(k): v for k, v in T.items()}
        for _ in range(iters):
            for k, v in keys.items():
                mm = cl == k; tot = wc[mm].sum()
                if tot > 0:
                    wc[mm] *= v / (tot / wc.sum())
            for col, tgt_ in extra.get(c, []):
                vals = D.loc[m, col].to_numpy(); tot_t = sum(v for k_, v in tgt_.items() if (vals == k_).any())
                for k_, v in tgt_.items():
                    mm = vals == k_
                    if mm.any() and wc[mm].sum() > 0:
                        wc[mm] *= (v / tot_t) / (wc[mm].sum() / wc.sum())
                rest = ~np.isin(vals, list(tgt_))
                if rest.any() and tot_t < 1:
                    wc[rest] *= (1 - tot_t) / max(wc[rest].sum() / wc.sum(), 1e-12)
            for p in np.unique(pans):
                mm = pans == p; wc[mm] *= (1.0 / len(np.unique(pans))) / (wc[mm].sum() / wc.sum())
        wc /= wc.sum(); w[m] = wc; ess[c] = float(1.0 / np.sum(wc ** 2))
    return w, ess


@functools.lru_cache(maxsize=None)
def _panel(p):
    P = load(p, "validation"); Qm = queue_matrix(P); q = (P.V <= P.vcut[None, :]) & np.isfinite(P.V)
    return P, Qm, q


_V3 = __import__("os").environ.get("TFB_M3VER", "") == "v3"   # 10/23 instrument v3 (G84): the replica tuned so the evidence-slot rows hidden
SIMN = int(__import__("os").environ.get("TFB_M3_SIMN", "10" if _V3 else "5"))     # (nothing observed or simulated blank) match the
SIMSP = int(__import__("os").environ.get("TFB_M3_SIMSP", "48" if _V3 else "72"))  # official 0.216: v2 5 / 72 -> 0.192, v3 10 / 48 -> 0.212


def scan_until(P, Qm, start, end, n_each=None, spacing=None, persist_max=0.9, cov_min=0.5):
    """t2_selector.scan restricted to origins in [start, end): same chronological rules (onset first-come with 6 h
    spacing; ongoing candidates then picked in order with spacing against onset + ongoing)"""
    from t2_windows import iou, H, F
    n_each = SIMN if n_each is None else n_each; spacing = SIMSP if spacing is None else spacing
    obs = np.isfinite(P.V) & P.E; any_q = Qm.any(axis=1); on, cands = [], []
    for t in range(max(start, H), min(end, P.T - F)):
        if P.times[t - H][:10] != P.times[t + F][:10] or obs[t - H:t].mean() < cov_min or not any_q[t + 1:t + F + 1].any():
            continue
        hq = Qm[t - H:t]; est = (hq.sum() >= 2) and (hq.sum(axis=0).max() >= 2)
        if not est and not hq.any() and not any_q[t]:
            if len(on) < n_each and all(abs(t - o) >= spacing for o in on):
                on.append(t)
        elif est and iou(np.broadcast_to(Qm[t - 1], (F, P.n)), Qm[t + 1:t + F + 1], P.E[t + 1:t + F + 1]) <= persist_max:
            cands.append(t)
    og = []
    for t in cands:
        if len(og) >= n_each:
            break
        if all(abs(t - o) >= spacing for o in on + og):
            og.append(t)
    return on, og


@functools.lru_cache(maxsize=None)
def sim_blank_rows(p, day, k, t_origin):
    """rows hidden by the selector replica run from the start of day `day - k` (k = 0: the window's own day) up to the
    origin t (exclusive)"""
    P, Qm, _ = _panel(p); start = (day - max(k, 0)) * 288
    on, og = scan_until(P, Qm, max(start, 12), t_origin)
    rows = set()
    for t in on + og:
        if t < t_origin:
            rows.update(range(t + 1, min(t + 19, t_origin)))
    return frozenset(rows)


def seen(p, t, k, lo, hi, cap=3):
    """links with an observed queue in rows dd+lo..dd+hi-1 on the earlier days back = 1..min(cap, k), with the
    selector-simulated blanks of those days hidden; returns (seen, number of days)"""
    P, _, q = _panel(p); s = np.zeros(P.n, bool); nd = 0; day = t // 288
    hid = sim_blank_rows(p, day, min(cap, k), t)
    for back in range(1, min(cap, k) + 1):
        dd = t - back * 288
        if dd + lo < 0:
            break
        rr = [r for r in range(dd + lo, dd + hi) if r not in hid]
        if rr:
            s |= q[rr].any(axis=0)
        nd += 1
    return s, nd


def summary(D, a, b, target="private", n_boot=2000, n_draw=4000, seed=0, seed_var=0.0, extra=None):
    """paired difference b - a: per-condition weighted means, T2 (mean of conditions), cluster (panel x day) bootstrap SE
    (+ seed_var added to the variance), P(B>A) over 80-window redraws (5 per panel x condition, within-cell weights),
    sign-flip permutation p (clusters), windows changed. D needs cond, panel, day, hour and the two columns."""
    rng = np.random.default_rng(seed); D = D.reset_index(drop=True); d = (D[b] - D[a]).to_numpy(); w, ess = weights(D, target, extra=extra)
    conds = [c for c in ("queue_onset", "queue_ongoing") if (D.cond == c).any()]
    per = {}
    for c in conds:
        m = (D.cond == c).to_numpy(); per[c] = dict(raw=float(d[m].mean()), weighted=float(np.sum(w[m] * d[m]) / w[m].sum()), n=int(m.sum()),
                                                   changed=int((np.abs(d[m]) > 1e-12).sum()), ess=round(ess[c], 1))
    t2 = float(np.mean([per[c]["weighted"] for c in conds]))
    cl = pd.factorize(D.panel.astype(str) + "_" + D.day.astype(str))[0]; ncl = cl.max() + 1
    cidx = np.array([conds.index(c) for c in D.cond]); M = np.zeros((ncl, len(conds))); Wm = np.zeros((ncl, len(conds)))
    np.add.at(M, (cl, cidx), w * d); np.add.at(Wm, (cl, cidx), w)
    R = rng.multinomial(ncl, np.full(ncl, 1.0 / ncl), size=n_boot)                     # bootstrap counts per cluster
    bs = np.mean((R @ M) / np.maximum(R @ Wm, 1e-12), axis=1)
    se = float(np.sqrt(np.var(bs) + seed_var))
    S = rng.choice([-1.0, 1.0], size=(n_boot, ncl))                                     # sign flips of cluster sums
    flips = np.mean((S @ M) / Wm.sum(axis=0), axis=1)
    obs = float(np.mean(M.sum(axis=0) / Wm.sum(axis=0)))
    p_perm = float(np.mean(np.abs(flips) >= abs(obs) - 1e-15))
    draws = np.zeros(n_draw)
    for c in conds:
        m = np.flatnonzero((D.cond == c).to_numpy()); acc = np.zeros(n_draw)
        pans = D.panel.to_numpy()[m]
        for p in np.unique(pans):
            mm = m[pans == p]; pr = w[mm] / w[mm].sum()
            acc += d[rng.choice(mm, size=(n_draw, 5), p=pr)].mean(axis=1) / len(np.unique(pans))
        draws += acc / len(conds)
    return dict(per_cond=per, t2_weighted=t2, se=se, p_b_gt_a=float(np.mean(draws > 0)), p_perm=p_perm, target=target)


def fmt(name, s):
    pc = " ".join(f"{c.split('_')[1]} w {v['weighted']:+.4f} raw {v['raw']:+.4f} (n {v['n']}, changed {v['changed']}, ESS {v['ess']})" for c, v in s["per_cond"].items())
    return (f"{name} [{s['target']}]: T2 {s['t2_weighted']:+.4f} (SE {s['se']:.4f}, {s['t2_weighted'] / max(s['se'], 1e-9):+.1f} SE), "
            f"P(B>A, 80 windows) {s['p_b_gt_a']:.2f}, perm p {s['p_perm']:.2f} | {pc}")
