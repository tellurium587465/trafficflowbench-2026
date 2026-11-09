"""G223 (11/03, ledger #189): read the dense X1 arms of g222 on M3 v3 DEV ongoing, A's flow of G133 (0.8 X1 + 0.2 LightGBM ->
cone -> seen x0.85 -> best_set -> far gate, X1 = the mean of c24f_attnglob s0 / s1). Main estimate = April composition
(m3v2.summary "private"), z/1.25 - E[max of 2]; March composition and raw beside it; odd / even days; the three thirds of
history coverage (nn_test_m3v3/R.npy[:, :12, 2], as G212). Identity: the base column == G133's base cell.
usage: TFB_M3VER=v3 py g223_x1_dense_read.py"""
import os, sys, pickle, importlib.util
os.environ.setdefault("TFB_M3VER", "v3")
import numpy as np, pandas as pd
R = r"C:/Users/user/tfb/"


def module(nm, path):
    spec = importlib.util.spec_from_file_location(nm, path); mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


g = module("g121", R + "review/g121_ongoing_oracle.py"); G, V, m3v2 = g.G, g.V, g.m3v2
ARMS = {"base": "", "d3e4": "_d3e4", "d1e1": "_d1e1"}; XS = {}
for c, arm in ARMS.items():
    fs = [g.W + f"probs_m3v3v2x1{arm}__c24f_attnglob_s{s}.pkl" for s in (0, 1)]
    if all(os.path.exists(f) for f in fs):
        a, b = (pickle.load(open(f, "rb")) for f in fs); XS[c] = {key: (a[key] + b[key]) / 2 for key in a}
print("arms", list(XS), flush=True)
cols = {c: {} for c in XS}
for w in g.DEV:
    m = G.META[w]; i = G.POS[w]; Xw = np.asarray(G.X[G.OFF[i]:G.OFF[i + 1]]); pl = g.FS.lgb_ongoing(Xw); kd = m3v2.kdist(m["day"]); cf = {k: g.cone_f(m, k) for k in kd}
    for c in XS:
        cols[c][w] = sum(pr * G.iou(g.og_pred((0.8 * XS[c][(w, k)] + 0.2 * pl) * cf[k], m, k)[0], m) for k, pr in kd.items())
D = V.frame(cols).dropna(subset=["base"]).reset_index(drop=True)
ref = pd.read_csv(g.W + "x1_grid/g133_x1_grid.csv").set_index("w")["l2e4_e4"]
print(f"{len(D)} windows | identity base vs G133 base cell max |d| {float(np.max(np.abs(D.base - ref.loc[D.w].to_numpy()))):.2e}", flush=True)
d_ = "C:/tfb_data/tfb_nn/nn_test_m3v3/"; R_ = np.load(d_ + "R.npy", mmap_mode="r"); o_ = np.load(d_ + "off.npy"); w_ = [str(x) for x in np.load(d_ + "window_id.npy", allow_pickle=True)]
COV = {x: float(np.asarray(R_[o_[i]:o_[i + 1], :12, 2], float).mean()) for i, x in enumerate(w_)}; D["cov"] = D.w.map(COV)
q1, q2 = D["cov"].quantile([1 / 3, 2 / 3]); D.to_csv(g.W + "g223_dev.csv", index=False)
EMAX2 = 0.5642
for c in XS:
    if c == "base":
        continue
    s = m3v2.summary(D, "base", c, "private"); sm = m3v2.summary(D, "base", c, "validation", n_boot=1000, n_draw=200); raw = float((D[c] - D.base).mean())
    z = s["t2_weighted"] / s["se"]; zc = z / 1.25 - EMAX2
    odd = D.day % 2 == 1; h = [m3v2.summary(D[k].reset_index(drop=True), "base", c, "private", n_boot=1000, n_draw=500)["t2_weighted"] for k in (odd, ~odd)]
    th = [m3v2.summary(D[k].reset_index(drop=True), "base", c, "private", n_boot=1000, n_draw=500) for k in (D["cov"] <= q1, (D["cov"] > q1) & (D["cov"] <= q2), D["cov"] > q2)]
    print(f"{c}: April composition {s['t2_weighted']:+.4f} (SE {s['se']:.4f}, z {z:+.2f}, z/1.25 - E[max of 2] {zc:+.2f}) | March composition {sm['t2_weighted']:+.4f} | raw {raw:+.4f} "
          f"| odd / even days {h[0]:+.4f} / {h[1]:+.4f} | coverage thirds (<= {q1:.3f}, <= {q2:.3f}, above) "
          + " / ".join(f"{t['t2_weighted']:+.4f} (SE {t['se']:.4f})" for t in th) + f" | line: corrected >= +1.64 {zc >= 1.64}, halves same sign {h[0] * h[1] > 0}", flush=True)
