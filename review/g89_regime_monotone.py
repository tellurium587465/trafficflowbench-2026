"""G89 (INSTRUMENT_V3_SPEC §2.1 revised (1), 10/23): monotonicity of the parts' effects in the March mask regime of the window's
day (the validation release has one Task 1 mask regime per calendar day; P.regime 0/1/2 = R1 20% / R2 30% / R3 50%
masked). Official windows' histories are unmasked, so a part whose M3 effect depends on the history coverage should move
monotonically from R3 (most masked) to R1 (least masked), towards its official-condition value. v3 windows outside the
hold-out; per regime the raw mean difference (B - A) and its SE (window bootstrap), and the observed history coverage.
usage: TFB_M3VER=v3 python g89_regime_monotone.py"""
import sys, os
os.environ.setdefault("TFB_M3VER", "v3")
sys.path.insert(0, r"C:/Users/user/tfb/src"); sys.argv = [sys.argv[0], "0"]
import numpy as np, pandas as pd
import t2_m3_gate as G, m3v2
from cachelib import load

W = G.W; V3 = G.VER
REG = {}
for w in G.WID:
    m = G.META[w]
    if m["panel"] not in REG:
        REG[m["panel"]] = load(m["panel"], "validation").regime
FR = {"gate": pd.read_csv(W + f"m3v2_gate{V3}.csv"), "parts": pd.read_csv(W + f"m3v2_parts{V3}.csv"), "x1": pd.read_csv(W + f"m3v2_x1{V3}.csv"),
      "rej": pd.read_csv(W + f"m3v2_rejudge{V3}.csv")}
PAIRS = [("evidence rule", "gate", "ev_none", "ev_cur", "queue_onset"), ("far gate", "gate", "gate_off", "gate_on", "queue_ongoing"),
         ("memory -> no memory", "gate", "mem", "nomem", None), ("cone V3", "parts", "cur", "cone", "queue_ongoing"),
         ("TTA", "parts", "cur", "tta", "queue_ongoing"), ("X1 ongoing", "x1", "base", "arm", "queue_ongoing"),
         ("physics onset adaptive", "rej", "nn_on", "physad", "queue_onset"), ("LightGBM ongoing", "rej", "nn_og", "lgb02", "queue_ongoing")]
rng = np.random.default_rng(0)
def se(x):
    x = np.asarray(x, float); return float(np.std([x[rng.integers(0, len(x), len(x))].mean() for _ in range(1000)])) if len(x) > 1 else float("nan")
rows = []
for name, f, a, b, c in PAIRS:
    D = FR[f].dropna(subset=[a, b]); D = D[~m3v2.is_holdout(D.day)]
    if c:
        D = D[D.cond == c]
    D = D.assign(regime=[int(REG[G.META[w]["panel"]][G.META[w]["t"]]) for w in D.w], cov=[G.META[w]["cov"] for w in D.w])
    r = dict(part=name)
    for rg in (0, 1, 2):
        g = D[D.regime == rg]; d = g[b] - g[a]
        r[f"R{rg + 1}"] = f"{d.mean():+.4f} ± {se(d):.4f} (n {len(g)}, cov {g['cov'].mean():.2f})"
    d1 = (D[D.regime == 0][b] - D[D.regime == 0][a]); d2 = (D[D.regime == 1][b] - D[D.regime == 1][a])
    r["R1 - R2"] = f"{d1.mean() - d2.mean():+.4f} ({(d1.mean() - d2.mean()) / max(np.hypot(se(d1), se(d2)), 1e-9):+.1f} SE)"   # R3 days rarely pass coverage >= 0.5
    rows.append(r)
print(pd.DataFrame(rows).to_string(index=False))
