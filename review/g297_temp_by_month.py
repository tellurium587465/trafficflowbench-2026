"""G297 (11/05, ledger #239 stage 0): does the best ongoing temperature move between months? From the dumps of G284 / G285 (the
probabilities just before the temperature, identities checked there), G286's cell log loss on the #234 grid, T* = the grid argmin, window
bootstrap 2000 for the 90 % interval. (1) G119's capB0 flow on the January and the February selector ongoing windows separately; (2) the
control: A17's March mix on M3 v3 dev ongoing, days 1-15 against 16 and later (private weights as G286). Line: |T*_Jan - T*_Feb| >= 0.2 with
non-overlapping 90 % intervals, and larger than the within-month first / second half difference. usage: TFB_M3VER=v3 py -B -X utf8 g297_temp_by_month.py"""
import os, sys, pickle
os.environ.setdefault("TFB_M3VER", "v3")
import numpy as np, pandas as pd
sys.path.insert(0, r"C:/Users/user/tfb/src")
import t2_m3_gate as G, m3v2
W = "C:/Users/user/tfb/work/t2/"; TS = np.array([0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.8, 2.0])
lg = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))


def tab(entries):
    S, N = [], []
    for y, msk, ks in entries:
        s_ = np.zeros(len(TS)); n_ = 0.0
        for pr, p in ks:
            z = lg(p[msk].astype(float)); yy = y[msk].astype(float); n_ += pr * msk.sum()
            for j, T in enumerate(TS):
                q = np.clip(1 / (1 + np.exp(-z / T)), 1e-7, 1 - 1e-7); s_[j] += pr * float(-(yy * np.log(q) + (1 - yy) * np.log(1 - q)).sum())
        S.append(s_); N.append(n_)
    return np.array(S), np.array(N)


def tstar(S, N, wt, rng, B=2000):
    f = lambda ix: TS[((wt[ix][:, None] * S[ix]).sum(axis=0) / (wt[ix] * N[ix]).sum()).argmin()]
    al = np.arange(len(S)); b = np.array([f(rng.integers(0, len(S), len(S))) for _ in range(B)]); return f(al), float(np.quantile(b, 0.05)), float(np.quantile(b, 0.95))


rng = np.random.default_rng(0); D5 = pickle.load(open(W + "g285_jf_pretemp.pkl", "rb")); R = {}
for mo in ("2031-01", "2031-02"):
    E = [(v["y"], v["e"], list(v["k"].values())) for v in D5.values() if v["month"] == mo]; S, N = tab(E); R[mo] = tstar(S, N, np.ones(len(S)), rng)
    print(f"JF capB0 {mo}: {len(E)} windows, T* {R[mo][0]} 90% [{R[mo][1]}, {R[mo][2]}]", flush=True)
D4 = pickle.load(open(W + "g284_og_pretemp_dev.pkl", "rb")); WID = sorted(D4)
FR = pd.DataFrame([dict(w=w, cond=G.META[w]["cond"], panel=G.META[w]["panel"], day=G.META[w]["day"], hour=(G.META[w]["t"] % 288) // 12) for w in WID]); WT = m3v2.weights(FR, "private")[0]
for lab, sel in (("days 1-15", FR.day.to_numpy() <= 14), ("days 16+", FR.day.to_numpy() >= 15)):
    ix = np.flatnonzero(sel); E = [(D4[WID[i]]["y"].T.astype(bool), D4[WID[i]]["m"].T.astype(bool), [(v[0], v[1]) for v in D4[WID[i]]["k"].values()]) for i in ix]
    S, N = tab(E); R[lab] = tstar(S, N, WT[ix], rng); print(f"M3 A17 ongoing {lab}: {len(ix)} windows, T* {R[lab][0]} 90% [{R[lab][1]}, {R[lab][2]}]", flush=True)
dm = abs(R["2031-01"][0] - R["2031-02"][0]); ov = not (R["2031-01"][2] < R["2031-02"][1] or R["2031-02"][2] < R["2031-01"][1]); dh = abs(R["days 1-15"][0] - R["days 16+"][0])
print(f"|T*_Jan - T*_Feb| {dm:.2f} (intervals overlap: {ov}) | within-month halves {dh:.2f} | line: {dm >= 0.2 and not ov and dm > dh}", flush=True)
