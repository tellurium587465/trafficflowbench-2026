"""G56: harm check of the X7 cone (G55 V1) on M1 = the train months' month-CV OOF (no shift), selector-like ongoing
windows, no-memory NN blend (the G42/G48 members). Same code path as the M3 test: t2_m3_gate.ongoing_pred (OGB x0.85 on
links unseen on up to 3 earlier days, expected-IoU set, far gate D=3 + contiguous growth) with the module's panel /
feature tables pointed at the train months (unmasked train cache, nn_data_afdem last_queue, no k limit, no trap 4).
The cone: g55_m3_x7_shock.parts (rows T-6..T-1, w = 25 km/h, x0.5). Paired difference, cluster bootstrap (panel x day).
usage: python g56_m1_x7_cone.py"""
import sys
sys.path.insert(0, r"C:/Users/user/tfb/src"); sys.path.insert(0, r"C:/Users/user/tfb/review")
sys.argv = [sys.argv[0], "0,1,2,3"]
import numpy as np, pandas as pd
import t2_m3_gate as G
from t2_eval_oof import T, oof_of
from cachelib import load, T2_PANELS
from common import NN
import importlib.util
spec = importlib.util.spec_from_file_location("g55", r"C:/Users/user/tfb/review/g55_m3_x7_shock.py")
src = open(spec.origin, encoding="utf-8").read().split("\nrows, st = [], []")[0]   # the definitions only, not the M3 run
g55 = type(sys)("g55"); exec(compile(src, spec.origin, "exec"), g55.__dict__)

MEM = ["q25_nomem_s3", "q25_nomem_s4", "q31_nomem_wp05_s0", "q31_nomem_attn1_s0", "q31_nomem_r13_s0", "q31_nomem_drop1_s0"]
CUR = np.array([1, 1, 2, 2, 1, 1], float); CUR /= CUR.sum()
O = [oof_of(m) for m in MEM]
keys = sorted(set.intersection(*[set(o) for o in O]) & {k for k in range(T.NWIN) if T.META["is_sel"][k] and str(T.META["cond"][k]) == "queue_ongoing"})
G.X = np.load(f"{NN}/nn_data_afdem/X.npy", mmap_mode="r"); G.OFF = np.load(f"{NN}/nn_data_afdem/off.npy")
rows = []
for p in T2_PANELS:
    kp = [k for k in keys if str(T.META["panel"][k]) == p]
    if not kp:
        continue
    P = load(p, "train"); G.PC.clear(); G.PC[p] = (P, (P.V <= P.vcut[None, :]) & np.isfinite(P.V)); G.EVENTS[p] = np.zeros(0, int)
    for k in kp:
        lab, el = T.LAB[k]; w = f"M1_{k}"; G.POS[w] = k
        m = dict(window_id=w, panel=p, t=int(T.META["t"][k]), cond="queue_ongoing", k=3, y=lab.T.astype(bool), m=el.T.astype(bool))
        G.META[w] = m
        pr = np.tensordot(CUR, np.stack([o[k] for o in O]), 1)
        S, cone, _ = g55.parts(w)
        rows.append(dict(cond="queue_ongoing", panel=p, cl=f"{p}_{str(T.META['date'][k])[:10]}", month=str(T.META["date"][k])[:7],
                         cur=G.iou(G.ongoing_pred(pr, m, True), m), cone=G.iou(G.ongoing_pred(pr * cone, m, True), m)))
    print(p, "windows", len(kp), flush=True)
    load.cache_clear()
D = pd.DataFrame(rows); D.to_csv(G.W + "g56_m1_x7_cone.csv", index=False); D["d"] = D.cone - D.cur
u = D.cl.unique(); idx = {x: np.flatnonzero(D.cl.to_numpy() == x) for x in u}; rng = np.random.default_rng(0)
se = np.std([D.d.to_numpy()[np.concatenate([idx[x] for x in rng.choice(u, len(u))])].mean() for _ in range(1000)])
print(f"M1 ongoing windows {len(D)}: current {D.cur.mean():.4f}, cone {D.cone.mean():.4f}, diff {D.d.mean():+.4f} (SE {se:.4f}, {D.d.mean() / max(se, 1e-9):+.1f} SE); "
      f"windows changed {int((D.d.abs() > 1e-9).sum())} (up {int((D.d > 1e-9).sum())}, down {int((D.d < -1e-9).sum())})")
print(D.groupby("panel").d.agg(["size", "mean"]).round(4).to_string())
print(D.groupby("month").d.mean().round(4).to_string())
