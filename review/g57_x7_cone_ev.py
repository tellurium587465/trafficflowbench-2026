"""G57: X7 cone, second try (STATE.md 10/22 pre-registration): V3 = the G55 cone (w = 25 km/h, x0.5) not applied to
links with same-month evidence (as the far gate: a queue observed the same day 05:00 .. T-1 or in the same slot T-90 ..
T+60 min on up to 3 earlier days, k-limited and trap 4 on M3, +-2 links). Scored on M3 (ongoing, production weights,
cluster SE, P(B>A)) and on M1 (train months' month-CV OOF, selector-like ongoing windows, cluster SE), same code path.
Pass: M3 >= +2 SE and M1 >= -1 SE.
usage: python g57_x7_cone_ev.py"""
import sys
sys.path.insert(0, r"C:/Users/user/tfb/src"); sys.path.insert(0, r"C:/Users/user/tfb/review")
sys.argv = [sys.argv[0], "0,1,2,3"]
import numpy as np, pandas as pd
import t2_m3_gate as G
from m3_stats import summary, fmt

src = open(r"C:/Users/user/tfb/review/g55_m3_x7_shock.py", encoding="utf-8").read().split("\nrows, st = [], []")[0]
g55 = type(sys)("g55"); exec(compile(src, "g55", "exec"), g55.__dict__)


def evidence(m):
    P, q = G.panel(m["panel"]); t = m["t"]; d0 = t - (t % 288) + 60
    ev = q[d0:t].any(axis=0) if t > d0 else np.zeros(P.n, bool)
    s2, _ = G.seen_rows(m["panel"], t, m["k"], -18, 12)
    return G.dil(ev | s2, 2)


def score(w, pr):
    m = G.META[w]; _, cone, _ = g55.parts(w); cev = np.where(evidence(m)[None, :], 1.0, cone)
    return G.iou(G.ongoing_pred(pr, m, True), m), G.iou(G.ongoing_pred(pr * cone, m, True), m), G.iou(G.ongoing_pred(pr * cev, m, True), m)


# M3
rows = []
for w in [w for w in G.WID if G.META[w]["cond"] == "queue_ongoing"]:
    m = G.META[w]; a, b, c = score(w, G.NOMEM[w])
    rows.append(dict(cond="queue_ongoing", panel=m["panel"], day=m["day"], cur=a, cone=b, cone_ev=c))
D3 = pd.DataFrame(rows)
print(fmt("M3 V1 cone (G55, check)", summary(D3, "cur", "cone")))
print(fmt("M3 V3 cone without evidence", summary(D3, "cur", "cone_ev")), "| windows changed", int(((D3.cone_ev - D3.cur).abs() > 1e-9).sum()))
print("   M3 V3 by panel:", D3.assign(d=D3.cone_ev - D3.cur).groupby("panel").d.mean().round(4).to_dict())

# M1 (module tables pointed at the train months, as g56)
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
        a, b, c = score(w, np.tensordot(CUR, np.stack([o[k] for o in O]), 1))
        rows.append(dict(panel=p, cl=f"{p}_{str(T.META['date'][k])[:10]}", cur=a, cone=b, cone_ev=c))
    load.cache_clear()
D1 = pd.DataFrame(rows); rng = np.random.default_rng(0)
u = D1.cl.unique(); idx = {x: np.flatnonzero(D1.cl.to_numpy() == x) for x in u}
for v in ("cone", "cone_ev"):
    d = (D1[v] - D1.cur).to_numpy(); se = np.std([d[np.concatenate([idx[x] for x in rng.choice(u, len(u))])].mean() for _ in range(1000)])
    print(f"M1 {v}: windows {len(D1)}, diff {d.mean():+.4f} (SE {se:.4f}, {d.mean() / max(se, 1e-9):+.1f} SE), changed {int((np.abs(d) > 1e-9).sum())} (up {int((d > 1e-9).sum())}, down {int((d < -1e-9).sum())})")
pd.concat([D3.assign(inst="M3"), D1.assign(inst="M1")]).to_csv(G.W + "g57_x7_cone_ev.csv", index=False)
