"""E3' (pre-registered STATE.md 10/22 10:35; the second try of E3): wider grid, and a "c only" control (beta = 0: a
uniform shift without the same-month observations), both chosen leave-one-panel-out on M3, ADD variant only (E3 then
the current evidence rule, then decode / far gate). Reports raw and production-weighted differences and P(B>A) over
80-window redraws (m3_stats), against the current rule and E3' against c-only.
usage: python t2_m3_e3b.py"""
import sys, itertools
import numpy as np, pandas as pd
import t2_m3_e3 as E                                     # loads M3, the q24f blends and the earlier-day statistics
G = E.G
from m3_stats import summary, fmt

GRID_FULL = list(itertools.product([0.25, 0.5, 1], [0.5, 1, 1.5, 2], [0, 0.5, 1]))
GRID_C = [(1, 0, c) for c in (0, 0.5, 1)]
rows = []
for cond in ("queue_onset", "queue_ongoing"):
    ws = [w for w in G.WID if G.META[w]["cond"] == cond]
    base = {w: G.iou(G.onset_pred(G.NOMEM[w], G.META[w], "cur") if cond == "queue_onset" else G.ongoing_pred(G.NOMEM[w], G.META[w], True), G.META[w]) for w in ws}
    res = {}
    for name, grid in (("full", GRID_FULL), ("conly", GRID_C)):
        score = {g: {w: G.iou(E.predict(w, G.META[w], *g, "ADD"), G.META[w]) for w in ws} for g in grid}
        chosen = {}
        for ph in sorted({G.META[w]["panel"] for w in ws}):
            tr = [w for w in ws if G.META[w]["panel"] != ph]
            chosen[ph] = max(grid, key=lambda g: np.mean([score[g][w] for w in tr]))
        res[name] = {w: score[chosen[G.META[w]["panel"]]][w] for w in ws}
        print(cond, name, "choices", sorted(set(chosen.values())), flush=True)
    for w in ws:
        m = G.META[w]
        rows.append(dict(cond=cond, panel=m["panel"], day=m["day"], band="0.5-0.6" if m["cov"] < 0.6 else "0.6-0.7",
                         base=base[w], full=res["full"][w], conly=res["conly"][w]))
D = pd.DataFrame(rows); D.to_csv(G.W + "m3_e3b.csv", index=False)
for cond in ("queue_onset", "queue_ongoing"):
    g = D[D.cond == cond]
    print(fmt(f"{cond} E3' vs current", summary(g, "base", "full")))
    print(fmt(f"{cond} c-only vs current", summary(g, "base", "conly")))
    print(fmt(f"{cond} E3' vs c-only", summary(g, "conly", "full")))
    for band, gb in g.groupby("band"):
        print(f"   band {band}: windows {len(gb)}, E3'-current {(gb.full - gb.base).mean():+.4f}, c-only-current {(gb.conly - gb.base).mean():+.4f}")
print(fmt("T2 E3' vs current", summary(D, "base", "full")))
