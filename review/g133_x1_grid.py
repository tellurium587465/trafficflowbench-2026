"""G133 (10/31, ledger #88, strategy): score the X1 lr x epochs grid on M3 v3 DEV ongoing, final A's flow (G121 path: 0.8 X1 + 0.2 LightGBM ->
cone -> seen x0.85 -> best_set -> far gate), X1 = the mean of c24f_attnglob s0 / s1. Chosen on odd days, read on even days.
usage: TFB_M3VER=v3 py g133_x1_grid.py"""
import os, sys, pickle, importlib.util, time
os.environ.setdefault("TFB_M3VER", "v3")
import numpy as np, pandas as pd
TM = time.perf_counter; T00 = TM(); R = r"C:/Users/user/tfb/"; OUT = R + "work/t2/x1_grid/"; os.makedirs(OUT, exist_ok=True)


def module(nm, path):
    spec = importlib.util.spec_from_file_location(nm, path); mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


def boot(d, cl, n_boot=2000, seed=0):
    rng = np.random.default_rng(seed); ids = pd.factorize(cl)[0]; M = np.bincount(ids, d); N = np.bincount(ids)
    Rm = rng.multinomial(len(M), np.full(len(M), 1.0 / len(M)), size=n_boot); return float(((Rm @ M) / (Rm @ N)).std())


g = module("g121", R + "review/g121_ongoing_oracle.py"); G, V, m3v2 = g.G, g.V, g.m3v2
CELLS = {"l2e4_e4": ""}
for lr in ("1e-4", "2e-4", "4e-4"):
    for ep in (2, 4, 8):
        if not (lr == "2e-4" and ep == 4):
            CELLS[f"l{lr.replace('-', '')}_e{ep}"] = f"_g{lr.replace('-', '')}e{ep}"
XS = {}
for c, arm in CELLS.items():
    fs = [g.W + f"probs_m3v3v2x1{arm}__c24f_attnglob_s{s}.pkl" for s in (0, 1)]
    if all(os.path.exists(f) for f in fs):
        a, b = (pickle.load(open(f, "rb")) for f in fs); XS[c] = {key: (a[key] + b[key]) / 2 for key in a}
print("cells", list(XS), flush=True)
cols = {c: {} for c in XS}
for w in g.DEV:
    m = G.META[w]; i = G.POS[w]; Xw = np.asarray(G.X[G.OFF[i]:G.OFF[i + 1]]); pl = g.FS.lgb_ongoing(Xw); kd = m3v2.kdist(m["day"]); cf = {k: g.cone_f(m, k) for k in kd}
    for c in XS:
        cols[c][w] = sum(pr * G.iou(g.og_pred((0.8 * XS[c][(w, k)] + 0.2 * pl) * cf[k], m, k)[0], m) for k, pr in kd.items())
D = V.frame(cols).dropna(subset=["l2e4_e4"]); ref = pd.read_csv(g.W + "g108_rectangles.csv").set_index("w")["A"]
print(f"{len(D)} windows in {TM() - T00:.0f}s | identity base cell vs G108 column A max |d| {float(np.max(np.abs(D.l2e4_e4 - ref.loc[D.w].to_numpy()))):.2e}")
D.to_csv(OUT + "g133_x1_grid.csv", index=False); cl = D.panel + "_" + D.day.astype(str); odd = (D.day % 2 == 1).to_numpy(); B = "l2e4_e4"
for c in XS:
    if c != B:
        x = (D[c] - D[B]).to_numpy(); se = boot(x, cl); print(f"{c}: dev {x.mean():+.4f} (SE {se:.4f}, z {x.mean() / max(se, 1e-12):+.2f}) | odd {x[odd].mean():+.4f} even {x[~odd].mean():+.4f}")
best = max((c for c in XS if c != B), key=lambda c: (D[c] - D[B])[odd].mean()); x = (D[best] - D[B]).to_numpy()[~odd]; se = boot(x, cl[~odd])
print(f"selected on odd days: {best}; even days {x.mean():+.4f} (SE {se:.4f}, z {x.mean() / max(se, 1e-12):+.2f})")
