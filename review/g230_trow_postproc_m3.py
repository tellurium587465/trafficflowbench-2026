"""G230 (11/03, ledger #197, the secondary read): the row at T in the post-processing on M3 v3 DEV ongoing. A's flow of G133 (0.8 X1 +
0.2 LightGBM -> cone -> seen x0.85 -> best_set -> far gate, X1 = c24f_attnglob s0 / s1, the saved files). The far gate (V.far_gate) and the
cone (g55.parts) read the last queue from the feature column G.CI; arm "trow" rewrites that column, in an in-memory copy, to the queue
state at the row T where M3's panel layer observes it (networks unchanged: their probabilities are saved). April composition, raw z
(fixed probabilities), odd / even days, the three thirds of history coverage (as G212). Identity: the base column == G133's base cell.
usage: TFB_M3VER=v3 py -B -X utf8 g230_trow_postproc_m3.py"""
import os, sys, pickle, importlib.util
os.environ.setdefault("TFB_M3VER", "v3")
import numpy as np, pandas as pd
R = r"C:/Users/user/tfb/"


def module(nm, path):
    spec = importlib.util.spec_from_file_location(nm, path); mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


g = module("g121", R + "review/g121_ongoing_oracle.py"); G, V, m3v2 = g.G, g.V, g.m3v2
a, b = (pickle.load(open(g.W + f"probs_m3v3v2x1__c24f_attnglob_s{s}.pkl", "rb")) for s in (0, 1)); X1 = {key: (a[key] + b[key]) / 2 for key in a}
X0 = np.array(G.X); XT = X0.copy(); nch = 0
for w in g.DEV:
    m = G.META[w]; i = G.POS[w]; P, _, q = m3v2._panel(m["panel"]); t = m["t"]; ot = np.isfinite(P.V[t]); lq = X0[G.OFF[i]:G.OFF[i + 1], G.CI] == 1
    lq2 = np.where(ot, q[t].astype(bool), lq); XT[G.OFF[i]:G.OFF[i + 1], G.CI] = lq2.astype(X0.dtype); nch += int((lq2 != lq).any())
cols = {"base": {}, "trow": {}}
for nm, XX in (("base", X0), ("trow", XT)):
    G.X = XX
    for w in g.DEV:
        m = G.META[w]; i = G.POS[w]; Xw = np.asarray(X0[G.OFF[i]:G.OFF[i + 1]]); pl = g.FS.lgb_ongoing(Xw); kd = m3v2.kdist(m["day"]); cf = {k: g.cone_f(m, k) for k in kd}
        cols[nm][w] = sum(pr * G.iou(g.og_pred((0.8 * X1[(w, k)] + 0.2 * pl) * cf[k], m, k)[0], m) for k, pr in kd.items())
G.X = X0
D = V.frame(cols).dropna(subset=["base"]).reset_index(drop=True)
ref = pd.read_csv(g.W + "x1_grid/g133_x1_grid.csv").set_index("w")["l2e4_e4"]
print(f"{len(D)} windows, last queue changed on {nch} | identity base vs G133 base cell max |d| {float(np.max(np.abs(D.base - ref.loc[D.w].to_numpy()))):.2e}", flush=True)
d_ = "C:/tfb_data/tfb_nn/nn_test_m3v3/"; R_ = np.load(d_ + "R.npy", mmap_mode="r"); o_ = np.load(d_ + "off.npy"); w_ = [str(x) for x in np.load(d_ + "window_id.npy", allow_pickle=True)]
COV = {x: float(np.asarray(R_[o_[i]:o_[i + 1], :12, 2], float).mean()) for i, x in enumerate(w_)}; D["cov"] = D.w.map(COV); q1, q2 = D["cov"].quantile([1 / 3, 2 / 3])
s = m3v2.summary(D, "base", "trow", "private"); sm = m3v2.summary(D, "base", "trow", "validation", n_boot=1000, n_draw=200); z = s["t2_weighted"] / s["se"]
odd = D.day % 2 == 1; h = [m3v2.summary(D[k].reset_index(drop=True), "base", "trow", "private", n_boot=1000, n_draw=500)["t2_weighted"] for k in (odd, ~odd)]
th = [m3v2.summary(D[k].reset_index(drop=True), "base", "trow", "private", n_boot=1000, n_draw=500) for k in (D["cov"] <= q1, (D["cov"] > q1) & (D["cov"] <= q2), D["cov"] > q2)]
print(f"trow - base: April composition {s['t2_weighted']:+.4f} (SE {s['se']:.4f}, z {z:+.2f}) | March composition {sm['t2_weighted']:+.4f} | raw {(D.trow - D.base).mean():+.4f} | odd / even {h[0]:+.4f} / {h[1]:+.4f} "
      f"| coverage thirds " + " / ".join(f"{t_['t2_weighted']:+.4f} (SE {t_['se']:.4f})" for t_ in th) + f" | windows changed {int((np.abs(D.trow - D.base) > 1e-12).sum())}")
D.to_csv(g.W + "g230_m3.csv", index=False)
