"""G86 (INSTRUMENT_V3_SPEC §6, 10/23): acceptance checks of the v3 windows from the scored frames (gate / parts / X1 on v3).
  V4 power : per pair, MDE(80%, two-sided 5%) = 2.8 x SE of the weighted T2 difference (April composition, main weights);
             spec line MDE <= 0.003
  V5 split : the pair's weighted difference on odd vs even March days; sign agreement among pairs with |z| > 1 (line >= 80%)
  V3 balance: standardised mean differences between the weighted M3 v3 windows and the official April windows for hour,
             weekend and set size (onset) / visible queue size (ongoing) is in m3v2_validate; here hour and weekend only,
             plus the ESS per condition (line >= 100)
usage: TFB_M3VER=v3 python g86_v3_accept.py"""
import sys, os
os.environ.setdefault("TFB_M3VER", "v3")
sys.path.insert(0, r"C:/Users/user/tfb/src"); sys.argv = [sys.argv[0], "0"]
import numpy as np, pandas as pd
import t2_m3_gate as G, m3v2

W = G.W; V3 = G.VER
FR = {"gate": pd.read_csv(W + f"m3v2_gate{V3}.csv"), "parts": pd.read_csv(W + f"m3v2_parts{V3}.csv"), "x1": pd.read_csv(W + f"m3v2_x1{V3}.csv")}
PAIRS = [("memory -> no memory", "gate", "mem", "nomem"), ("evidence rule", "gate", "ev_none", "ev_cur"), ("far gate", "gate", "gate_off", "gate_on"),
         ("sim ongoing", "gate", "nomem", "sim"), ("B1 skip", "gate", "ev_cur", "skip"), ("negative control", "gate", "nomem", "neg"),
         ("TTA", "parts", "cur", "tta"), ("cone V3", "parts", "cur", "cone"), ("EMA", "parts", "ema0", "ema1"), ("X1 ongoing", "x1", "base", "arm")]
rows = []
for name, f, a, b in PAIRS:
    D = FR[f].dropna(subset=[a, b]); D = D[D.cond == "queue_ongoing"] if name == "X1 ongoing" else D
    s = m3v2.summary(D, a, b, "private", n_boot=500, n_draw=200)
    halves = [m3v2.summary(D[D.day % 2 == h], a, b, "private", n_boot=200, n_draw=50)["t2_weighted"] for h in (0, 1)]
    rows.append(dict(pair=name, d=s["t2_weighted"], se=s["se"], z=s["t2_weighted"] / s["se"], mde=2.8 * s["se"], odd=halves[1], even=halves[0],
                     agree=bool(np.sign(halves[0]) == np.sign(halves[1]))))
R = pd.DataFrame(rows); print(R.round(4).to_string(index=False))
big = R[R.z.abs() > 1]
print(f"V4: MDE <= 0.003 in {int((R.mde <= 0.003).sum())}/{len(R)} pairs (median MDE {R.mde.median():.4f})")
print(f"V5: halves agree in {int(big.agree.sum())}/{len(big)} pairs with |z| > 1 ({big.agree.mean() if len(big) else float('nan'):.0%}; line 80%)")
_, ess = m3v2.weights(FR["gate"], "private"); print("V3: ESS", {c: round(v) for c, v in ess.items()}, "(line 100)")
