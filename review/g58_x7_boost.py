"""G58: X7 V2' (STATE.md 10/22 pre-registration): the G55 shock extrapolation used only to RAISE probabilities inside the
extrapolated run (head .. L(T) + s h upstream): p -> 1 - (1 - p) * 0.5 there, untouched elsewhere (G55 V2 averaged a 0/1
map with the NN and so halved every other cell). Ongoing windows only; same post-processing as the M3 gate. Scored on
M3 (production weights, cluster SE, P(B>A)) and M1 (train months' month-CV OOF, as g56/g57), with the edge decomposition
of G52 (FN growth under-predicted / FP queue that should vanish kept / detached) for the current and the V2' sets.
Pass: M3 >= +2 SE and P(B>A) >= 0.8, and M1 >= -1 SE.
usage: python g58_x7_boost.py"""
import sys
sys.path.insert(0, r"C:/Users/user/tfb/src"); sys.path.insert(0, r"C:/Users/user/tfb/review")
sys.argv = [sys.argv[0], "0,1,2,3"]
import numpy as np, pandas as pd
import t2_m3_gate as G
from m3_stats import summary, fmt

src = open(r"C:/Users/user/tfb/review/g55_m3_x7_shock.py", encoding="utf-8").read().split("\nrows, st = [], []")[0]
g55 = type(sys)("g55"); exec(compile(src, "g55", "exec"), g55.__dict__)


def edge(sel, m, lq):
    """cells (observed) of FN next to / away from the visible queue and FP next to / away from it"""
    y = m["y"].T.astype(bool) & m["m"].T.astype(bool); s = sel & m["m"].T.astype(bool); out = np.zeros(4, int)
    for r in range(s.shape[0]):
        fn = y[r] & ~s[r]; fp = s[r] & ~y[r]; nfn = g55.touch(y[r], lq, len(lq)) if hasattr(g55, "touch") else None
        near = G.dil(lq, 1)
        out += [int((fn & near).sum()), int((fn & ~near).sum()), int((fp & near).sum()), int((fp & ~near).sum())]
    return out


def score(w, pr):
    m = G.META[w]; S, _, _ = g55.parts(w); pb = np.where(S > 0, 1 - (1 - pr) * 0.5, pr)
    a = G.ongoing_pred(pr, m, True); b = G.ongoing_pred(pb, m, True)
    i = G.POS[w]; lq = np.asarray(G.X[G.OFF[i]:G.OFF[i + 1], G.CI], dtype=float) == 1
    return G.iou(a, m), G.iou(b, m), edge(a, m, lq), edge(b, m, lq)


rows, ea, eb = [], np.zeros(4, int), np.zeros(4, int)
for w in [w for w in G.WID if G.META[w]["cond"] == "queue_ongoing"]:
    m = G.META[w]; a, b, xa, xb = score(w, G.NOMEM[w]); ea += xa; eb += xb
    rows.append(dict(cond="queue_ongoing", panel=m["panel"], day=m["day"], cur=a, boost=b))
D3 = pd.DataFrame(rows)
print(fmt("M3 V2' boost inside the extrapolated run", summary(D3, "cur", "boost")), "| windows changed", int(((D3.boost - D3.cur).abs() > 1e-9).sum()))
print("   M3 by panel:", D3.assign(d=D3.boost - D3.cur).groupby("panel").d.mean().round(4).to_dict())
print("   M3 edge cells [FN near (growth under), FN far (new missed), FP near (kept), FP far (spurious)]: current", ea.tolist(), "-> V2'", eb.tolist())

from t2_eval_oof import T, oof_of
from cachelib import load, T2_PANELS
from common import NN
MEM = ["q25_nomem_s3", "q25_nomem_s4", "q31_nomem_wp05_s0", "q31_nomem_attn1_s0", "q31_nomem_r13_s0", "q31_nomem_drop1_s0"]
CUR = np.array([1, 1, 2, 2, 1, 1], float); CUR /= CUR.sum(); O = [oof_of(m) for m in MEM]
keys = sorted(set.intersection(*[set(o) for o in O]) & {k for k in range(T.NWIN) if T.META["is_sel"][k] and str(T.META["cond"][k]) == "queue_ongoing"})
G.X = np.load(f"{NN}/nn_data_afdem/X.npy", mmap_mode="r"); G.OFF = np.load(f"{NN}/nn_data_afdem/off.npy")
rows = []
for p in T2_PANELS:
    kp = [k for k in keys if str(T.META["panel"][k]) == p]
    P = load(p, "train"); G.PC.clear(); G.PC[p] = (P, (P.V <= P.vcut[None, :]) & np.isfinite(P.V)); G.EVENTS[p] = np.zeros(0, int)
    for k in kp:
        lab, el = T.LAB[k]; w = f"M1_{k}"; G.POS[w] = k
        G.META[w] = dict(window_id=w, panel=p, t=int(T.META["t"][k]), cond="queue_ongoing", k=3, y=lab.T.astype(bool), m=el.T.astype(bool))
        a, b, _, _ = score(w, np.tensordot(CUR, np.stack([o[k] for o in O]), 1))
        rows.append(dict(panel=p, cl=f"{p}_{str(T.META['date'][k])[:10]}", cur=a, boost=b))
    load.cache_clear()
D1 = pd.DataFrame(rows); rng = np.random.default_rng(0); d = (D1.boost - D1.cur).to_numpy()
u = D1.cl.unique(); idx = {x: np.flatnonzero(D1.cl.to_numpy() == x) for x in u}
se = np.std([d[np.concatenate([idx[x] for x in rng.choice(u, len(u))])].mean() for _ in range(1000)])
print(f"M1 V2': windows {len(D1)}, diff {d.mean():+.4f} (SE {se:.4f}, {d.mean() / max(se, 1e-9):+.1f} SE), changed {int((np.abs(d) > 1e-9).sum())} (up {int((d > 1e-9).sum())}, down {int((d < -1e-9).sum())})")
pd.concat([D3.assign(inst="M3"), D1.assign(inst="M1")]).to_csv(G.W + "g58_x7_boost.csv", index=False)
