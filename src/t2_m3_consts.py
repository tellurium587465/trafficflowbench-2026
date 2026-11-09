"""April-window post-processing constants on M3 (pre-registered STATE.md 10/22 10:55), leave-one-panel-out.
onset  : evidence factor B {0.1, .25, .5, 1}, evidence-day cap {3, 6}, width {1, 2, 3}, Platt (G49 a=-0.396, b=0.640) {no, yes}
ongoing: evidence factor OGB {0.7, .85, 1}, evidence-day cap {3, 6}, far gate D {2, 3, 4}, CONTIG {0, 1}
The constants are chosen on the other panels and scored on the held-out panel; compared with the current constants
(raw, production-weighted, P(B>A) over 80-window redraws). q24f seeds 0-3, k-day limit and trap 4 as t2_m3_gate.
usage: python t2_m3_consts.py"""
import sys, itertools
sys.argv = [sys.argv[0], "0,1,2,3"]
import numpy as np, pandas as pd
import t2_m3_gate as G
from m3_stats import summary, fmt

PLATT = (-0.3963, 0.6402)
lg = lambda x: np.log(np.clip(x, 1e-5, 1 - 1e-5) / (1 - np.clip(x, 1e-5, 1 - 1e-5)))


def onset_c(p_, m, B, cap, width, platt):
    pb = p_.copy()
    if platt:
        pb = np.where(pb > 0, 1 / (1 + np.exp(-(PLATT[0] + PLATT[1] * lg(pb)))), 0.0)
    if B < 1:
        s, nd = G.seen_rows(m["panel"], m["t"], m["k"], -6, 12, cap=cap)
        if nd:
            pb[-1] = pb[-1] * np.where(G.dil(s, width), 1.0, B)
    return G.best_set(pb.ravel(), np.ones(pb.size, bool)).reshape(pb.shape)


def far_gate_c(sel, m, cap, D, contig):
    P, q = G.panel(m["panel"]); t = m["t"]; d0 = t - (t % 288) + 60
    ev = q[d0:t].any(axis=0) if t > d0 else np.zeros(P.n, bool)
    s2, _ = G.seen_rows(m["panel"], t, m["k"], -18, 12, cap=cap)
    ev = G.dil(ev | s2, 2)
    i = G.POS[m["window_id"]]; lq = np.asarray(G.X[G.OFF[i]:G.OFF[i + 1], G.CI], dtype=float) == 1
    near = np.zeros(P.n, bool)
    for j in np.flatnonzero(lq):
        near[max(0, j - D):j + D + 1] = True
    C = np.zeros_like(sel)
    if contig:
        for r_ in range(sel.shape[0]):
            j = 0
            while j < sel.shape[1]:
                if sel[r_, j]:
                    e = j
                    while e + 1 < sel.shape[1] and sel[r_, e + 1]:
                        e += 1
                    if near[j:e + 1].any():
                        C[r_, j:e + 1] = True
                    j = e + 1
                else:
                    j += 1
    return sel & ~(sel & ~near[None, :] & ~ev[None, :] & ~C)


def ongoing_c(p_, m, OGB, cap, D, contig):
    pb = p_.copy()
    if OGB < 1:
        s, nd = G.seen_rows(m["panel"], m["t"], m["k"], -18, 12, cap=cap)
        if nd:
            pb = p_ * np.where(G.dil(s, 1), 1.0, OGB)[None, :]
    sel = G.best_set(pb.ravel(), np.ones(pb.size, bool)).reshape(pb.shape)
    return far_gate_c(sel, m, cap, D, contig)


SPACE = {"queue_onset": (list(itertools.product([0.1, 0.25, 0.5, 1.0], [3, 6], [1, 2, 3], [0, 1])), (0.25, 3, 2, 0), onset_c),
         "queue_ongoing": (list(itertools.product([0.7, 0.85, 1.0], [3, 6], [2, 3, 4], [0, 1])), (0.85, 3, 3, 1), ongoing_c)}
rows = []
for cond, (grid, cur, fn) in SPACE.items():
    ws = [w for w in G.WID if G.META[w]["cond"] == cond]
    score = {g: {w: G.iou(fn(G.NOMEM[w], G.META[w], *g), G.META[w]) for w in ws} for g in grid}
    assert abs(np.mean([score[cur][w] for w in ws]) - np.mean([G.iou(G.onset_pred(G.NOMEM[w], G.META[w], "cur") if cond == "queue_onset" else G.ongoing_pred(G.NOMEM[w], G.META[w], True), G.META[w]) for w in ws])) < 1e-12, "current constants do not reproduce the gate"
    chosen = {}
    for ph in sorted({G.META[w]["panel"] for w in ws}):
        tr = [w for w in ws if G.META[w]["panel"] != ph]
        chosen[ph] = max(grid, key=lambda g: np.mean([score[g][w] for w in tr]))
    full = max(grid, key=lambda g: np.mean([score[g][w] for w in ws]))
    print(cond, "current", cur, "| chosen per held-out panel", sorted(set(chosen.values())), "| best on all panels", full,
          f"(in-sample {np.mean([score[full][w] for w in ws]):.4f} vs current {np.mean([score[cur][w] for w in ws]):.4f})", flush=True)
    for w in ws:
        m = G.META[w]
        rows.append(dict(cond=cond, panel=m["panel"], day=m["day"], band="0.5-0.6" if m["cov"] < 0.6 else "0.6-0.7",
                         cur=score[cur][w], nested=score[chosen[m["panel"]]][w]))
D = pd.DataFrame(rows); D.to_csv(G.W + "m3_consts.csv", index=False)
for cond in ("queue_onset", "queue_ongoing"):
    g = D[D.cond == cond]; print(fmt(f"{cond} nested constants vs current", summary(g, "cur", "nested")))
    for band, gb in g.groupby("band"):
        print(f"   band {band}: windows {len(gb)}, nested-current {(gb.nested - gb.cur).mean():+.4f}")
