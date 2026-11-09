"""E3' (ongoing, ADD) and the onset shift (E3' c-only) re-scored under M3 v2 (strategy §11 (b)). For each M3 window and
each k of the official distribution (truncated at the window's day; E3 itself is not capped, the evidence rule inside is
capped at 3), the earlier-day statistics use the pseudo-origins b = 1..k (t2_m3v2_prev_build.py) with the
selector-simulated blanks of (day, k) hidden; logit p' = logit p + beta (logit pi - logit pbar) + c where observed
(as t2_m3_e3.corrected); then the current rule / decoder / far gate under v2; the window's score is the expectation over
k. (a, beta, c) chosen leave-one-family-out (D7_I10, D7_I210, D7_I405, D12_I5; review D M4) by the v2-weighted mean.
usage: python t2_m3v2_e3.py"""
import sys, itertools, pickle
sys.argv = [sys.argv[0], "0,1,2,3"]
import numpy as np, pandas as pd
import t2_m3_gate as G
import t2_m3v2_gate as V
import m3v2

lg = lambda x: np.log(x / (1 - x))
clip = lambda x: np.clip(x, 1e-4, 1 - 1e-4)
PM = pickle.load(open(G.W + f"m3{G.VER}prev6_meta.pkl", "rb"))   # 10/23: per window-set version
parts = [(w, pickle.load(open(G.W + f"probs_m3{G.VER}prev6__{f}_s{s}.pkl", "rb"))) for s in G.SEEDS for f, w in G.CFG5("q24f")]
tw = sum(w for w, _ in parts); PP = {k: sum(w * pr[k] for w, pr in parts) / tw for k in parts[0][1]}
BYP = {}
for e in PM:
    BYP.setdefault(e["parent"], []).append(e)
_ST = {}


def stat(w, k):
    """(k_q, k_o, sum pbar) over b <= k with the simulated blanks of (day, k) hidden"""
    key = (w, k)
    if key not in _ST:
        m = G.META[w]; hid = m3v2.sim_blank_rows(m["panel"], m["day"], k, m["t"]); kq = ko = sp = 0.0
        for e in BYP.get(w, []):
            if e["b"] > k:
                continue
            keep = np.array([r not in hid for r in range(e["tp"] + 1, e["tp"] + 7)])[None, :]   # (1, 6) rows of this origin
            mo = e["m"].astype(float) * keep; kq = kq + e["y"].astype(float) * keep; ko = ko + mo; sp = sp + PP[e["window_id"]].T * mo
        _ST[key] = (kq, ko, sp)
    return _ST[key]


def corrected(p_, w, k, a, beta, c):
    if beta == 0 and c == 0:
        return p_
    kq, ko, sp = stat(w, k)
    if np.isscalar(ko):
        return p_
    kq, ko, sp = kq.T, ko.T, sp.T; has = ko > 0
    pbar = clip(np.where(has, sp / np.maximum(ko, 1), 0.5)); pi = clip((kq + a * pbar) / (ko + a))
    out = 1 / (1 + np.exp(-(lg(clip(p_)) + np.where(has, beta * (lg(pi) - lg(pbar)) + c, 0.0))))
    return np.where(p_ > 0, out, 0.0)


def score(w, g):
    m = G.META[w]
    def one(k):
        p_ = corrected(G.NOMEM[w], w, k, *g)
        sel = V.onset_pred(p_, m, "cur", k) if m["cond"] == "queue_onset" else V.ongoing_pred(p_, m, True, k)
        return G.iou(sel, m)
    return sum(pr * one(k) for k, pr in m3v2.kdist(m["day"]).items())


FAM = lambda p: p.rsplit("_", 1)[0]
GRID = {"queue_onset": [(1, 0, c) for c in (0, 0.5, 1)], "queue_ongoing": list(itertools.product([0.25, 0.5, 1], [0.5, 1, 1.5, 2], [0, 0.5, 1]))}
base = V.frame({"x": {}})
cols = {"cur": {}, "e3": {}}
for cond, grid in GRID.items():
    ws = [w for w in G.WID if G.META[w]["cond"] == cond]
    sc = {g: {w: score(w, g) for w in ws} for g in grid + [(1, 0, 0)]}
    Dc = V.frame({str(g): sc[g] for g in sc}); Dc = Dc[Dc.cond == cond].reset_index(drop=True); wts, _ = m3v2.weights(Dc, "private")
    fam = Dc.panel.map(FAM).to_numpy()
    dev = ~m3v2.is_holdout(Dc.day.to_numpy())                 # 10/23: constants chosen on the windows outside the hold-out only
    for f in np.unique(fam):
        tr = (fam != f) & dev
        best = max(grid, key=lambda g: np.sum(wts[tr] * Dc[str(g)].to_numpy()[tr]) / wts[tr].sum())
        for w in Dc.w[~tr]:
            cols["e3"][w] = sc[best][w]
        print(cond, "held-out family", f, "chose", best, flush=True)
    for w in ws:
        cols["cur"][w] = sc[(1, 0, 0)][w]
    best_all = max(grid, key=lambda g: np.sum(wts[dev] * Dc[str(g)].to_numpy()[dev]) / wts[dev].sum())
    print(cond, "production constants (all families, dev windows):", best_all, flush=True)
D = V.frame(cols); D.to_csv(G.W + f"m3v2_e3{G.VER}.csv", index=False)
for cond, name in (("queue_onset", "onset shift (E3' c-only)"), ("queue_ongoing", "E3' ADD (ongoing)")):
    for tgt in ("private", "validation"):
        print(m3v2.fmt(name, m3v2.summary(D[D.cond == cond], "cur", "e3", tgt)))
