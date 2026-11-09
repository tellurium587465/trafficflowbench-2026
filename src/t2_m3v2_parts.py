"""Re-score today's adopted parts under M3 v2 (strategy session §11 (b): one instrument version for all parts; rule
unchanged: drop a part only if its weighted point estimate <= 0 or some instrument shows harm <= -2 SE). This script
covers the parts that need no earlier-day statistics: TTA (both conditions), cone V3 (ongoing; evidence exemption with
the v2 hidden rows), EMA (e4s_aema vs e4s_a). E3' / the onset shift / X1 need rebuilt earlier-day inputs (separate).
Both target mixes are printed (private = April, the main one; validation = March, for the public LB).
usage: python t2_m3v2_parts.py"""
import sys
sys.argv = [sys.argv[0], "0,1,2,3"]
import pickle, numpy as np, pandas as pd
import t2_m3_gate as G
import t2_m3v2_gate as V
import m3v2

lg = lambda x: np.log(np.clip(x, 1e-6, 1 - 1e-6) / (1 - np.clip(x, 1e-6, 1 - 1e-6)))


def blend(prefix):
    parts = [(w, pickle.load(open(G.W + f"probs_{prefix}__{f}_s{s}.pkl", "rb"))) for f, w in G.CFG5("q24f") for s in G.SEEDS]
    tw = sum(w for w, _ in parts)
    return {k: sum(w * pr[k] for w, pr in parts) / tw for k in parts[0][1]}


def mean2(tags):
    ps = [pickle.load(open(G.W + f"probs_{G.PRE}__{t}.pkl", "rb")) for t in tags]
    return {k: sum(p[k] for p in ps) / len(ps) for k in ps[0]}


draws = [blend(f"{G.PRE}tta{k}") for k in range(1, 9)]          # 10/23: per window-set version (G.PRE)
TTA = {}
for w in G.WID:
    st = np.stack([d[w] for d in draws]); TTA[w] = np.where((st == 0).all(axis=0), 0.0, 1 / (1 + np.exp(-lg(st).mean(axis=0))))
src = open(r"C:/Users/user/tfb/review/g55_m3_x7_shock.py", encoding="utf-8").read().split("\nrows, st = [], []")[0]
g55 = type(sys)("g55"); exec(compile(src, "g55", "exec"), g55.__dict__)
E0, E1 = mean2(["e4s_a_s0", "e4s_a_s1"]), mean2(["e4s_aema_s0", "e4s_aema_s1"])


def cone_score(p_, m):
    _, cone, _ = g55.parts(m["window_id"])
    def one(k):
        P, _, q = m3v2._panel(m["panel"]); t = m["t"]; d0 = t - (t % 288) + 60; hid = V._hidden_today(m, k)
        rr = [r for r in range(d0, t) if r not in hid]
        ev = q[rr].any(axis=0) if rr else np.zeros(P.n, bool)
        s2, _ = m3v2.seen(m["panel"], t, k, -18, 12)
        f = np.where(G.dil(ev | s2, 2)[None, :], 1.0, cone)
        return G.iou(V.ongoing_pred(p_ * f, m, True, k), m)
    return V.expect(m, one)


cols = {c: {} for c in ("cur", "tta", "cone", "tta_cone", "ema0", "ema1")}
for w in G.WID:
    m = G.META[w]
    cols["cur"][w] = V.score(G.NOMEM[w], m); cols["tta"][w] = V.score(TTA[w], m)
    cols["ema0"][w] = V.score(E0[w], m); cols["ema1"][w] = V.score(E1[w], m)
    if m["cond"] == "queue_ongoing":
        cols["cone"][w] = cone_score(G.NOMEM[w], m); cols["tta_cone"][w] = cone_score(TTA[w], m)
    else:
        cols["cone"][w] = cols["cur"][w]; cols["tta_cone"][w] = cols["tta"][w]
D = V.frame(cols); D.to_csv(G.W + f"m3v2_parts{G.VER}.csv", index=False)
for name, a, b in (("TTA", "cur", "tta"), ("cone V3", "cur", "cone"), ("March set (TTA + cone V3)", "cur", "tta_cone"), ("EMA", "ema0", "ema1")):
    for tgt in ("private", "validation"):
        print(m3v2.fmt(name, m3v2.summary(D, a, b, tgt)))
