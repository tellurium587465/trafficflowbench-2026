"""Combination check of the parts adopted under the 10/22 decision rule (STATE.md 10/22 pre-registration), on M3.
Parts: TTA input (8 perturbed feature sets, logit mean; t2_m3_tta.py), E3' prior correction (t2_m3_e3.corrected; onset:
the c-only grid, ongoing: the full grid; (a, beta, c) re-chosen leave-one-panel-out inside each combination), cone V3
(G55 cone x0.5 except links with same-month evidence; ongoing only). Then the current evidence rule / decoder / far gate.
  April set : TTA + E3' + cone V3          March set : TTA + cone V3
with one-part-removed variants; every variant against the current pipeline (NOMEM, current post-processing).
usage: python t2_m3_combo.py"""
import sys, itertools, pickle
sys.argv = [sys.argv[0], "0,1,2,3"]
import numpy as np, pandas as pd
import t2_m3_e3 as E
G = E.G
from m3_stats import summary, fmt

lg = lambda x: np.log(np.clip(x, 1e-6, 1 - 1e-6) / (1 - np.clip(x, 1e-6, 1 - 1e-6)))


def blend(prefix):
    parts = [(w, pickle.load(open(G.W + f"probs_{prefix}__{f}_s{s}.pkl", "rb"))) for f, w in G.CFG5("q24f") for s in G.SEEDS]
    tw = sum(w for w, _ in parts)
    return {k: sum(w * pr[k] for w, pr in parts) / tw for k in parts[0][1]}


draws = [blend(f"m3tta{k}") for k in range(1, 9)]
TTA = {}
for w in G.WID:
    st = np.stack([d[w] for d in draws]); TTA[w] = np.where((st == 0).all(axis=0), 0.0, 1 / (1 + np.exp(-lg(st).mean(axis=0))))
src = open(r"C:/Users/user/tfb/review/g55_m3_x7_shock.py", encoding="utf-8").read().split("\nrows, st = [], []")[0]
g55 = type(sys)("g55"); exec(compile(src, "g55", "exec"), g55.__dict__)


def evidence(m):
    P, q = G.panel(m["panel"]); t = m["t"]; d0 = t - (t % 288) + 60
    ev = q[d0:t].any(axis=0) if t > d0 else np.zeros(P.n, bool)
    s2, _ = G.seen_rows(m["panel"], t, m["k"], -18, 12)
    return G.dil(ev | s2, 2)


CONE = {}
for w in G.WID:
    if G.META[w]["cond"] == "queue_ongoing":
        _, cone, _ = g55.parts(w); CONE[w] = np.where(evidence(G.META[w])[None, :], 1.0, cone)
GRID = {"queue_onset": [(1, 0, c) for c in (0, 0.5, 1)],
        "queue_ongoing": list(itertools.product([0.25, 0.5, 1], [0.5, 1, 1.5, 2], [0, 0.5, 1]))}


def run(w, tta, e3, cone, g=None):
    m = G.META[w]; p = TTA[w] if tta else G.NOMEM[w]
    if e3:
        p = E.corrected(p, w, *g)
    if m["cond"] == "queue_onset":
        return G.iou(G.onset_pred(p, m, "cur"), m)
    if cone:
        p = p * CONE[w]
    return G.iou(G.ongoing_pred(p, m, True), m)


CONF = {"April: TTA+E3'+cone": (1, 1, 1), "April without TTA": (0, 1, 1), "April without E3'": (1, 0, 1), "April without cone": (1, 1, 0),
        "March: TTA+cone": (1, 0, 1), "TTA only": (1, 0, 0), "cone only": (0, 0, 1)}
D = pd.DataFrame([dict(w=w, cond=G.META[w]["cond"], panel=G.META[w]["panel"], day=G.META[w]["day"], cur=run(w, 0, 0, 0)) for w in G.WID])
for name, (tta, e3, cone) in CONF.items():
    col = {}
    for cond in ("queue_onset", "queue_ongoing"):
        ws = [w for w in G.WID if G.META[w]["cond"] == cond]
        if not e3:
            col.update({w: run(w, tta, 0, cone) for w in ws}); continue
        score = {g: {w: run(w, tta, 1, cone, g) for w in ws} for g in GRID[cond]}
        for ph in sorted({G.META[w]["panel"] for w in ws}):
            tr = [w for w in ws if G.META[w]["panel"] != ph]; g = max(GRID[cond], key=lambda g_: np.mean([score[g_][w] for w in tr]))
            col.update({w: score[g][w] for w in ws if G.META[w]["panel"] == ph})
    D[name] = D.w.map(col)
    s = summary(D, "cur", name); print(fmt(f"{name} vs current", s), flush=True)
D.to_csv(G.W + "m3_combo.csv", index=False)
