"""G64 (10/22 night, user: reopen the rare-site onset): what do the lost onset windows look like on M3 (dev windows only,
hold-out excluded)? Production onset blend (s46 spec, q24f), raw T+30 probabilities before the evidence rule.
Rarity f as G54 (train months, panel x weekend x 2-hour slot, rare = f <= 0.1). Per onset window with an observed
queued cell at T+30: IoU of the production set (k expectation), whether the truth holds a rare link, the rank of the best
true link in the raw probabilities, p(best true) / p(top), link distance from the top link to the nearest true link,
and the true set size.
usage: python g64_onset_rare_diag.py"""
import sys, json, pickle
sys.path.insert(0, r"C:/Users/user/tfb/review")
exec(open(r"C:/Users/user/tfb/review/g54_m3_onset_rare.py", encoding="utf-8").read().split("# link frequency f from the train months")[0])
import numpy as np, pandas as pd
import t2_m3v2_gate as V, m3v2

# rarity f (same code as G54)
CNT, DEN = {}, {}
for k in SELON:
    lab, el = T.LAB[k]; y = (lab[5] & el[5]).astype(float)
    key = tkey(T.META["panel"][k], pd.Timestamp(str(T.META["date"][k])[:10]).dayofweek >= 5, T.META["t"][k])
    CNT[key] = CNT.get(key, 0) + y; DEN[key] = DEN.get(key, 0) + 1
def mkey(m): return tkey(m["panel"], ((MARCH_FIRST_DOW + m["day"]) % 7) >= 5, m["t"])
def freq(m):
    return CNT[mkey(m)] / DEN[mkey(m)] if mkey(m) in DEN else np.zeros(len(m["y"]))

spec = json.load(open(r"C:/Users/user/tfb/src/t2_specs.json"))["s37"]["spec"]
parts = [(x.split(":")[0], float(x.split(":")[1]) if ":" in x else 1.0) for x in spec.split("+")]
PR = {t: pickle.load(open(G.W + f"probs_m3__{t}.pkl", "rb")) for t, _ in parts}
rows = []
for w in G.WID:
    m = G.META[w]
    if m["cond"] != "queue_onset" or m3v2.is_holdout(m["day"]):
        continue
    y = m["y"][:, 5].astype(bool) & m["m"][:, 5].astype(bool)
    if not y.any():
        continue
    p = sum(wt * PR[t][w] for t, wt in parts) / sum(wt for _, wt in parts); p30 = p[5]
    f = freq(m); order = np.argsort(-p30); rank = {int(i): r for r, i in enumerate(order)}
    tl = np.flatnonzero(y); best = tl[np.argmax(p30[tl])]
    iou = V.score(p, m)
    rows.append(dict(w=w, panel=m["panel"], day=m["day"], iou=iou, n_true=int(y.sum()), rare_true=bool((f[tl] <= 0.1).all()),
                     any_rare=bool((f[tl] <= 0.1).any()), rank_best=rank[int(best)], ratio=float(p30[best] / p30[order[0]]),
                     dist=int(np.min(np.abs(tl - order[0]))), p_top=float(p30[order[0]]), f_top=float(f[order[0]]),
                     f_true=float(f[best])))
D = pd.DataFrame(rows); D.to_csv(G.W + "g64_onset_rare_diag.csv", index=False)
print(f"dev onset windows with a T+30 queue: {len(D)}; IoU mean {D.iou.mean():.3f}; lost (IoU < 0.2): {(D.iou < 0.2).sum()}")
for name, g in (("all-rare truth", D[D.rare_true]), ("common truth", D[~D.rare_true])):
    print(f"{name}: n {len(g)}, IoU {g.iou.mean():.3f}, rank of best true link: median {g.rank_best.median():.0f}, "
          f"top-1 {np.mean(g.rank_best == 0):.2f}, top-3 {np.mean(g.rank_best < 3):.2f}, top-5 {np.mean(g.rank_best < 5):.2f}; "
          f"p(true)/p(top) median {g.ratio.median():.3f}; distance top->true: 0 {np.mean(g.dist == 0):.2f}, <=2 {np.mean(g.dist <= 2):.2f}, "
          f"<=5 {np.mean(g.dist <= 5):.2f}; f(top) median {g.f_top.median():.2f}, p(top) median {g.p_top.median():.3f}")
L = D[D.iou < 0.2]
print(f"lost windows: n {len(L)}, all-rare truth {L.rare_true.mean():.2f}, best-true rank median {L.rank_best.median():.0f} "
      f"(top-3 {np.mean(L.rank_best < 3):.2f}), distance <=2 {np.mean(L.dist <= 2):.2f}, n_true median {L.n_true.median():.0f}, "
      f"p(top) median {L.p_top.median():.3f}, f(top) median {L.f_top.median():.2f}")
print(L.sort_values("rank_best")[["w", "iou", "n_true", "rare_true", "rank_best", "ratio", "dist", "p_top", "f_top", "f_true"]].round(3).to_string(index=False))
