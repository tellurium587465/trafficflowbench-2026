"""G294 (11/05, ledger #237 stage 0 (b)): B7f on M3 v3 dev ongoing 356 windows and the pair's E[max]. Per (window, k): A17 = G284's dumped
mix at temperature 0.8 -> best_set -> G132's far gate (G287's T 0.8 column); B7 = the raw best_set of the q24f CF5 8-seed blend (G276's B
column, k-free); B7f = B7 with its split cells (0.35 <= p_B7 <= 0.65) flipped to the opposite of A17 where A17 = B7 on them, when there
are >= 3 such cells. Identities: A17 and B7 against G287 / G276. E[max] as G292 (emax_table, (1) as they are, (2) centred), onset from
G291 (B7f's onset = B7's). Line: both point > 0 and z >= +1. usage: TFB_M3VER=v3 py -B -X utf8 g294_b7f_m3.py"""
import os, sys, pickle, importlib.util
os.environ.setdefault("TFB_M3VER", "v3")
import numpy as np, pandas as pd
R = r"C:/Users/user/tfb/"; sys.path.insert(0, R + "src"); sys.path.insert(0, R + "review")
import t2_m3_gate as G
from t2_model import best_set
from emax import emax_table
W = R + "work/t2/"; CF5 = (("attnglob", 2.0), ("attnwp05", 2.0), ("attn1", 2.0), ("drop1", 1.0), ("r13", 1.0)); ON = [(f"q24f_{c}_s{s}", w) for s in range(8) for c, w in CF5]
lg = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6))); tmp = lambda p, T: 1 / (1 + np.exp(-lg(p) / T))
bs = lambda pb: best_set(pb.ravel(), np.ones(pb.size, bool)).reshape(pb.shape)


def module(nm, path, argv):
    a = sys.argv; sys.argv = argv; spec = importlib.util.spec_from_file_location(nm, path); mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); sys.argv = a
    return mod


g132 = module("g132", R + "review/g132_p3_grid.py", ["g132", "run"])
D4 = pickle.load(open(W + "g284_og_pretemp_dev.pkl", "rb")); PQ = {t: pickle.load(open(W + f"probs_m3v3__{t}.pkl", "rb")) for t, _ in ON}; WS = sum(w for _, w in ON); rows = []; nfl = 0
for w, v in D4.items():
    m = G.META[w]; pq = sum(wt * PQ[t][w] for t, wt in ON) / WS; b = bs(pq); sp = (pq >= 0.35) & (pq <= 0.65); r = dict(w=w, A=0.0, B=G.iou(b, m), Bf=0.0, flipped=0.0)
    for pr, p, ex, nr, _ in v["k"].values():
        a = g132.gate(bs(tmp(p, 0.8)), ex, nr); ag = sp & (a == b); f = b.copy()
        if ag.sum() >= 3:
            f[ag] = ~a[ag]; r["flipped"] += pr
        r["A"] += pr * G.iou(a, m); r["Bf"] += pr * G.iou(f, m)
    nfl += r["flipped"] > 0; rows.append(r)
og = pd.DataFrame(rows); og.to_csv(W + "g294_m3.csv", index=False)
r7 = pd.read_csv(W + "g287_m3.csv").set_index("w")["t0.8"]; r6 = pd.read_csv(W + "g276_dev.csv").set_index("w").B
print(f"M3 dev ongoing {len(og)} | identity A17 against g287 t0.8: {float(np.abs(og.set_index('w').A - r7.reindex(og.w).to_numpy()).max()):.1e}, B7 against g276 B: {float(np.abs(og.set_index('w').B - r6.reindex(og.w).to_numpy()).max()):.1e} | "
      f"windows flipped (some k) {nfl}; B7f - B7 {(og.Bf - og.B).mean():+.5f} | A17 {og.A.mean():.4f} B7 {og.B.mean():.4f} B7f {og.Bf.mean():.4f}", flush=True)
on = pd.read_csv(W + "g291_m3.csv")[["w", "pc4", "a0"]].rename(columns={"pc4": "A", "a0": "B"}); on["Bf"] = on["B"]
C = {"A17": ("A", "A"), "B7": ("B", "B"), "B7f": ("Bf", "Bf")}; P = [("A17", "B7f"), ("A17", "B7")]
for lab, sh in (("(1) as they are", False), ("(2) centred", True)):
    o, g = on.copy(), og.copy()
    if sh:
        do, dg = (o.A - o.B).mean(), (g.A - g.B).mean(); o["B"] += do; o["Bf"] += do; g["B"] += dg; g["Bf"] += dg
    T = emax_table(o, g, C, P, ("A17", "B7")); lv, d, se = T[("A17", "B7f")]
    print(f"{lab}: E[max] (A17, B7) {T[('A17', 'B7')][0]:.5f}, (A17, B7f) {lv:.5f}; gain {d:+.6f} (SE {se:.6f}, z {d / max(se, 1e-12):+.2f}) | line (point > 0 and z >= +1): {d > 0 and d / max(se, 1e-12) >= 1}", flush=True)
