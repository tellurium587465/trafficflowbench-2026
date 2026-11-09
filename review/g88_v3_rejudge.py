"""G88 (INSTRUMENT_V3_SPEC §9, 10/23): re-judge the adopted parts on the v3 windows (main weights, outside the hold-out,
src/judge.py). This script builds the v3 frame for the parts not covered by m3v2_gatev3 / m3v2_partsv3 / m3v2_x1v3:
  phys03 / physad : physics onset (physon_base, the real train months) mixed at T+30, a = 0.3 / the causal month-adaptive
                    a (t2_phys_onset.adaptive_a on the March events), before the evidence rule; onset = s46 blend (q24f)
  lgb02           : the no-memory LightGBM (lgbnm models) mixed at a = 0.2 into the ongoing production blend (s70 spec)
  e5              : the onset blend over q24f seeds 0-7 vs 0-3
Writes work/t2/m3v2_rejudgev3.csv (columns nn_on, phys03, physad, nn_og, lgb02, s4, s8).
usage: TFB_M3VER=v3 python g88_v3_rejudge.py"""
import os, sys, json, pickle
os.environ.setdefault("TFB_M3VER", "v3")
sys.path.insert(0, r"C:/Users/user/tfb/src"); sys.argv = [sys.argv[0], "0"]
import numpy as np, pandas as pd, lightgbm as lgb
import t2_m3_gate as G, t2_m3v2_gate as V, m3v2
import t2_phys_onset as PO
from common import NN

W = G.W; PRE = G.PRE
spec_on = json.load(open(r"C:/Users/user/tfb/src/t2_specs.json"))["s37"]["spec"]
spec_og = "+".join(f"{f}_{c}_s{s}{w}" for s in range(4) for f in ("q43f", "q46f") for c, w in (("attnglob", ":2"), ("attnwp05", ":2"), ("attn1", ":2"), ("drop1", ""), ("r13", "")))
_C = {}
def pr(t):
    if t not in _C:
        _C[t] = pickle.load(open(W + f"probs_{PRE}__{t}.pkl", "rb"))
    return _C[t]
def blend(spec, w):
    parts = [(x.split(":")[0], float(x.split(":")[1]) if ":" in x else 1.0) for x in spec.split("+")]
    return sum(wt * pr(m)[w] for m, wt in parts) / sum(wt for _, wt in parts)
# physics onset on the v3 windows and the month-adaptive a (March events before each window's day)
X = np.load(f"{NN}/nn_test_m3{G.VER}/X.npy", mmap_mode="r"); OFF = np.load(f"{NN}/nn_test_m3{G.VER}/off.npy")
phys = lgb.Booster(model_file=W + "physon_base.txt")
ev = PO.month_events("validation", PO.rarity_table())
# no-memory LightGBM on the v3 windows (the cache column layout = the NN's, checked 10/22)
os.environ.update(TFB_T2_NOT="1", TFB_T2CACHE="2_noT_af", TFB_T2_ARR="1", TFB_T2_DEM="1", TFB_T2_DROP="prior_on,prior_og,pos,vf,capl,lanes,length,w,tod,dow")
import t2_fast as TF
LG = {c: lgb.Booster(model_file=W + f"lgbnm_{c}.txt") for c in ("queue_onset", "queue_ongoing")}
s4 = [(w_, f"{fam}_s{s}") for fam, w_ in G.CFG5("q24f") for s in range(4)]; s8 = [(w_, f"{fam}_s{s}") for fam, w_ in G.CFG5("q24f") for s in range(8)]
cols = {c: {} for c in ("nn_on", "phys03", "physad", "nn_og", "lgb02", "s4", "s8")}
for i, w in enumerate(G.WID):
    m = G.META[w]; Xw = np.asarray(X[OFF[i]:OFF[i + 1]])
    cols["s4"][w] = V.score(sum(a * pr(t)[w] for a, t in s4) / sum(a for a, _ in s4), m)
    cols["s8"][w] = V.score(sum(a * pr(t)[w] for a, t in s8) / sum(a for a, _ in s8), m)
    if m["cond"] == "queue_onset":
        p = blend(spec_on, w); ph = phys.predict(PO.feats(Xw)); cols["nn_on"][w] = V.score(p, m)
        for name, a in (("phys03", 0.3), ("physad", PO.adaptive_a(ev, m["day"]))):
            q = p.copy(); q[5] = (1 - a) * p[5] + a * ph; cols[name][w] = V.score(q, m)
    else:
        p = blend(spec_og, w); cols["nn_og"][w] = V.score(p, m)
        Xd = Xw.copy(); Xd[:, TF.I_PON] = 0.0; Xd[:, TF.I_POG] = 0.0; Xd = TF.apply_drop(Xd)
        pl = LG["queue_ongoing"].predict(np.vstack([TF.step_block(Xd, s + 1) for s in range(6)]), num_threads=8).reshape(6, len(Xd))
        cols["lgb02"][w] = V.score(0.8 * p + 0.2 * pl, m)
D = V.frame(cols); D.to_csv(W + f"m3v2_rejudge{G.VER}.csv", index=False); print("wrote", len(D), "windows")
