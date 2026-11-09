"""G357 (11/05, ledger #281 stage 0, measurement only): G334's box-level calibration split by the window origin's clock band (T's hour
in the data's timestamps: morning <= 9, midday 10-15, evening 16-17, night >= 18). Items = (window, k draw, box = last-queue run, horizon row):
s = mean probability over the box's links after the temperature 0.8, o = share of the box's scored cells queued, weight = the k weight.
M3 = A17's mix (G284 dump), January / February = capB0 (G285 dump), June official 40 ongoing windows = cvf (G347's dump, the k draw of the
largest weight, weight 1). Per instrument x band: n, ECE (deciles), mean s - mean o (> 0 = over-confident that the box stays).
Stage 1 trigger (#281): evening or night ECE >= 0.05 with the same sign of (s - o) on all three. usage: TFB_M3VER=v3 py -B -X utf8 g357_box_calib_tod.py"""
import os, sys, pickle
os.environ.setdefault("TFB_M3VER", "v3")
import numpy as np, pandas as pd
R0 = r"C:/Users/user/tfb/"; sys.path.insert(0, R0 + "src")
import t2_m3_gate as G
W = R0 + "work/t2/"; lg = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6))); tmp = lambda p, T: 1 / (1 + np.exp(-lg(p) / T))
band = lambda t: (lambda h: "morning" if h <= 9 else "midday" if h <= 15 else "evening" if h <= 17 else "night")((int(t) % 288) // 12)


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


def boxes(src, b_, lq, y, e, ks, temp=True):
    out = []
    for a, b in runs(lq):
        for pr, p in ks:
            q = tmp(p, 0.8) if temp else p
            for r in range(6):
                sc = e[r, a:b + 1]
                if sc.any():
                    out.append(dict(src=src, band=b_, row=r + 1, s=float(q[r, a:b + 1].mean()), o=float(y[r, a:b + 1][sc].mean()), w=pr))
    return out


rows = []
for w, v in pickle.load(open(W + "g284_og_pretemp_dev.pkl", "rb")).items():
    m = G.META[w]; i_ = G.POS[w]; lq = np.asarray(G.X[G.OFF[i_]:G.OFF[i_ + 1], G.CI], float) == 1
    rows += boxes("M3", band(m["t"]), lq, m["y"].T.astype(bool), m["m"].T.astype(bool), [(x[0], x[1]) for x in v["k"].values()])
NN = "C:/tfb_data/tfb_nn/nn_data_afdem/"; X = np.load(NN + "X.npy", mmap_mode="r"); off = np.load(NN + "off.npy"); tt = np.load(NN + "t.npy")
for k, v in pickle.load(open(W + "g285_jf_pretemp.pkl", "rb")).items():
    lq = np.asarray(X[off[k]:off[k + 1], G.CI], float) == 1; rows += boxes("JF", band(tt[k]), lq, v["y"], v["e"], list(v["k"].values()))
A = pickle.load(open(W + "g342_jun80.pkl", "rb"))
for k, v in pickle.load(open(W + "g347_og_jun80.pkl", "rb")).items():
    if isinstance(k, str) or k < A["NWIN"]:
        continue
    pr, kk, q, st, lq, y, e, p, t, day, split = v; rows += boxes("JUN", band(t), lq.astype(bool), y.astype(bool), e.astype(bool), [(1.0, q)], temp=False)
D = pd.DataFrame(rows); D.to_csv(W + "g357_box_calib_tod.csv", index=False)


def ece(d):
    d = d.copy(); nb = min(10, max(2, len(d) // 20)); d["bin"] = pd.qcut(d.s.rank(method="first"), nb, labels=False)
    g = d.groupby("bin").apply(lambda x: pd.Series(dict(n=x.w.sum(), s=np.average(x.s, weights=x.w), o=np.average(x.o, weights=x.w))))
    return float((g.n * (g.s - g.o).abs()).sum() / g.n.sum())


out = []
for (s_, b_), d in D.groupby(["src", "band"]):
    out.append(dict(src=s_, band=b_, items=len(d), ece=ece(d) if len(d) >= 40 else np.nan, s_minus_o=float(np.average(d.s - d.o, weights=d.w))))
T = pd.DataFrame(out); print(T.pivot(index="band", columns="src", values=["items", "ece", "s_minus_o"]).round(3).to_string(), flush=True)
trig = []
for b_ in ("evening", "night"):
    t_ = T[T.band == b_]
    if len(t_) == 3 and (t_.ece >= 0.05).any() and len(set(np.sign(t_.s_minus_o))) == 1:
        trig.append(b_)
print("stage 1 trigger (evening / night ECE >= 0.05 with the same sign on all three):", trig or "none", flush=True)
