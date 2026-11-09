"""G334 (11/05, ledger #265 stage 0, measurement only): box-level calibration. Boxes = the last-queue runs; per (window, k draw, box, horizon
row r = 1 .. 6) the score s = the mean probability over the box's links at row r after A17's temperature 0.8 (what the decode sees; M3 = A17's
mix of G284, January / February = capB0 of G285), the outcome = the share of the box's scored cells queued at row r (rows without scored cells
left out), weight = the k weight. Per row and over all rows: deciles of s, mean s against the mean outcome, ECE = sum n_b / N |mean s - mean
outcome|. Line (#265): ECE >= 0.03 -> stage 1 (isotonic per row, fitted on one instrument, read on the other).
usage: TFB_M3VER=v3 py -B -X utf8 g334_box_calib.py"""
import os, sys, pickle
os.environ.setdefault("TFB_M3VER", "v3")
import numpy as np, pandas as pd
R0 = r"C:/Users/user/tfb/"; sys.path.insert(0, R0 + "src")
import t2_m3_gate as G
W = R0 + "work/t2/"; lg = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6))); tmp = lambda p, T: 1 / (1 + np.exp(-lg(p) / T))


def runs(m):
    out, j = [], 0
    while j < len(m):
        if m[j]:
            e_ = j
            while e_ + 1 < len(m) and m[e_ + 1]:
                e_ += 1
            out.append((j, e_)); j = e_ + 1
        else:
            j += 1
    return out


def boxes(src, lq, y, e, ks):
    out = []
    for a, b in runs(lq):
        for pr, p in ks:
            q = tmp(p, 0.8)
            for r in range(6):
                sc = e[r, a:b + 1]
                if sc.any():
                    out.append(dict(src=src, row=r + 1, s=float(q[r, a:b + 1].mean()), o=float(y[r, a:b + 1][sc].mean()), w=pr))
    return out


rows = []
D4 = pickle.load(open(W + "g284_og_pretemp_dev.pkl", "rb"))
for w, v in D4.items():
    m = G.META[w]; i_ = G.POS[w]; lq = np.asarray(G.X[G.OFF[i_]:G.OFF[i_ + 1], G.CI], float) == 1
    rows += boxes("M3", lq, m["y"].T.astype(bool), m["m"].T.astype(bool), [(x[0], x[1]) for x in v["k"].values()])
NN = "C:/tfb_data/tfb_nn/nn_data_afdem/"; X = np.load(NN + "X.npy", mmap_mode="r"); off = np.load(NN + "off.npy"); CI = G.CI
D5 = pickle.load(open(W + "g285_jf_pretemp.pkl", "rb"))
for k, v in D5.items():
    lq = np.asarray(X[off[k]:off[k + 1], CI], float) == 1; rows += boxes("JF", lq, v["y"], v["e"], list(v["k"].values()))
D = pd.DataFrame(rows); D.to_csv(W + "g334_box_calib.csv", index=False)


def ece(d):
    d = d.copy(); d["bin"] = pd.qcut(d.s.rank(method="first"), 10, labels=False); g = d.groupby("bin").apply(lambda x: pd.Series(dict(n=x.w.sum(), s=np.average(x.s, weights=x.w), o=np.average(x.o, weights=x.w))))
    return float((g.n * (g.s - g.o).abs()).sum() / g.n.sum()), g


for s_ in ("M3", "JF"):
    d = D[D.src == s_]; e_all, g = ece(d)
    print(f"--- {s_}: {len(d)} (box, row, k) items; ECE over all rows {e_all:.4f} | deciles mean s -> mean outcome: " + ", ".join(f"{a:.2f}->{b:.2f}" for a, b in zip(g.s, g.o)), flush=True)
    print("    per row ECE: " + ", ".join(f"r{r} {ece(d[d.row == r])[0]:.4f}" for r in range(1, 7)), flush=True)
print(f"line (ECE >= 0.03 on either instrument -> stage 1): {max(ece(D[D.src == s_])[0] for s_ in ('M3', 'JF')) >= 0.03}", flush=True)
