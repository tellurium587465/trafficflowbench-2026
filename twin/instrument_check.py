"""Is the law-calibrated twin (gen3 L worlds) a better LB stand-in than the old test worlds T? The LB measured the effect
of adding randomized twin worlds to training: ongoing IoU about +0.025 (s67 vs s60: +0.0038 / 0.15), onset about
-0.017 (s64 vs s63: -0.0026 / 0.15). Same comparison inside the twin: A-trained NN (tw2_nomem s0+s1) vs A+R-trained
(tw2ar24 + tw2ar48t2, s0+s1), production post-processing (onset x0.25 rule; ongoing x0.85 + far gate + contiguous
growth), on the T worlds (GPU probs) and the L worlds (CPU re-inference, t2_nn_cpu_infer.py). Cluster bootstrap over
panel x world. usage: python instrument_check.py"""
import sys, glob, pickle, numpy as np, pandas as pd, warnings
warnings.simplefilter("ignore")
sys.path.insert(0, r"C:/Users/user/tfb/twin")
from adjuster import bestset, world, gate, rule_probs
W = "C:/Users/user/tfb/work/t2/"


def blend(paths):
    P = [pickle.load(open(p, "rb")) for p in paths]
    return {k: sum(p[k] for p in P) / len(P) for k in P[0]}


def score(LT, PA, PS):
    rows = []
    for wid, rec in LT.items():
        if wid not in PA:
            continue
        y, el, panel, wd, t, cond, date = rec
        Y = y.T.astype(bool); EL = el.T.astype(bool)
        if cond == "queue_ongoing":
            P_, obs, full = world(panel, wd); fut = Y & EL; pp = np.broadcast_to(full[t - 1], fut.shape) & EL   # review 10/13: the selector looks at t-1; u = (pp | fut).sum()
            if u and (pp & fut).sum() / u > 0.9:
                continue
        def iou(s):
            uu = ((s | Y) & EL).sum(); return 1.0 if uu == 0 else (s & Y & EL).sum() / uu
        if cond == "queue_onset" and (t + 6) % 288 >= 144:
            continue                                                   # review 10/13: official onset windows are mornings
        r = dict(cond=cond, cl=panel + wd)
        for nm, pr in (("A", PA[wid]), ("S", PS[wid])):
            p = pr.copy()
            if cond == "queue_onset":
                p[:-1] = 0
            r[nm] = iou(gate(bestset(rule_probs(rec, p)), rec)) if cond == "queue_ongoing" else iou(bestset(rule_probs(rec, p)))
        rows.append(r)
    return pd.DataFrame(rows)


def report(name, R):
    rng = np.random.default_rng(0)
    for c, g in R.groupby("cond"):
        d = (g.S - g.A).to_numpy(); cl = g.cl.to_numpy(); u = np.unique(cl); idx = {x: np.flatnonzero(cl == x) for x in u}
        se = np.std([d[np.concatenate([idx[x] for x in rng.choice(u, len(u))])].mean() for _ in range(400)])
        print(f"{name:3s} {c:13s} n {len(g):5d}: A-trained {g.A.mean():.4f}, +R {g.S.mean():.4f}, diff {d.mean():+.4f} (SE {se:.4f})")


LT = pickle.load(open("E:/tfb_nn/twin_labels.pkl", "rb"))
report("T", score(LT, blend([W + f"probs_tw2_nomem_s{s}.pkl" for s in (0, 1)]),
                  blend([W + f"probs_{m}_nomem_s{s}.pkl" for m in ("tw2ar24",) for s in (0, 1)])))   # review 10/13: 2 vs 2 models
LL = {}
for f in glob.glob("E:/tfb_nn/twin_labels_twL_*.pkl"):
    LL.update(pickle.load(open(f, "rb")))
report("L", score(LL, blend([W + f"probs_twL__tw2_nomem_s{s}.pkl" for s in (0, 1)]),
                  blend([W + f"probs_twL__{m}_nomem_s{s}.pkl" for m in ("tw2ar24",) for s in (0, 1)])))   # review 10/13: 2 vs 2 models
print("LB reference: ongoing about +0.025, onset about -0.017 (IoU units)")
