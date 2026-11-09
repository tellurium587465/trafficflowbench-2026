"""G292 (11/05, ledger #236 stage 0 (b)): E[max] of the pair (A17, B7s) against (A17, B7) on M3 v3 dev, emax.emax_table as G251 / #214
(40 onset + 40 ongoing windows per draw, S = 0.30 T2, outer bootstrap 200 x inner 200). Onset: G291 (pc4 = A17, a0 = B7, a0n = B7s,
100 windows); ongoing: A17 = G287's T 0.8 column, B7 = G276's B column (raw best_set of q24f 8 seeds, no X1 / X3 / temperature), 356
windows, B7s = B7 there. (1) the per-window values as they are; (2) B7 and B7s shifted per condition by the mean A17 - B7 (centred).
Line: both (1) and (2) point > 0 and z >= +1. usage: py -B -X utf8 g292_b7s_emax.py"""
import sys
import numpy as np, pandas as pd
sys.path.insert(0, r"C:/Users/user/tfb/review"); from emax import emax_table
W = "C:/Users/user/tfb/work/t2/"
on = pd.read_csv(W + "g291_m3.csv")[["w", "pc4", "a0", "a0n"]].rename(columns={"pc4": "A", "a0": "B", "a0n": "Bs"})
og = pd.read_csv(W + "g287_m3.csv")[["w", "t0.8"]].rename(columns={"t0.8": "A"}).merge(pd.read_csv(W + "g276_dev.csv")[["w", "B"]], on="w"); og["Bs"] = og["B"]
assert len(on) == 100 and len(og) == 356
C = {"A17": ("A", "A"), "B7": ("B", "B"), "B7s": ("Bs", "Bs")}; P = [("A17", "B7s"), ("A17", "B7")]
for lab, sh in (("(1) as they are", False), ("(2) centred", True)):
    o, g = on.copy(), og.copy()
    if sh:
        do, dg = (o.A - o.B).mean(), (g.A - g.B).mean(); o["B"] += do; o["Bs"] += do; g["B"] += dg; g["Bs"] += dg
    T = emax_table(o, g, C, P, ("A17", "B7")); lv, d, se = T[("A17", "B7s")]
    print(f"{lab}: E[max] (A17, B7) {T[('A17', 'B7')][0]:.5f}, (A17, B7s) {lv:.5f}; gain {d:+.6f} (SE {se:.6f}, z {d / max(se, 1e-12):+.2f}) | line (point > 0 and z >= +1): {d > 0 and d / max(se, 1e-12) >= 1}", flush=True)
print(f"means: onset A17 {on.A.mean():.4f} B7 {on.B.mean():.4f} B7s {on.Bs.mean():.4f} | ongoing A17 {og.A.mean():.4f} B7 {og.B.mean():.4f}", flush=True)
