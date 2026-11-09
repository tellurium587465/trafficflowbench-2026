"""G286 (11/05, ledger #234 as revised before the computation): the ongoing temperature chosen by cell-level proper scores.
Inputs (dumped just before the temperature, identities checked there): G284 = M3 v3 dev 356 ongoing windows, A's final March mix (A16) and
G119's capB0 flow; G285 = the 316 January / February selector ongoing windows, G119's capB0 flow. Per (window, k) with the k weight, the
scored cells (M3: m["m"]; Jan / Feb: the eligibility). p_T = sigmoid(logit(p) / T), T on the declared grid. Main: the mean cell log loss;
secondary: Brier, and the log loss on the cells with 0.02 <= p <= 0.98 (pre-temperature). M3 windows carry the private-composition
weights of m3v2.weights (unweighted also printed); Jan / Feb unweighted. T* = the grid argmin; window bootstrap 2000 (M3 windows jointly
for A and capB0, Jan / Feb independently): 90 % intervals of T*_M3(A), delta = T*_JF(capB0) - T*_M3(capB0), and T*_M3(A) + delta.
Line (revised): both T*_M3(A) and T*_M3(A) + delta have 90 % intervals that exclude 1.2 on the same side -> the candidate is the end nearer
1.2 (0.05 steps); otherwise 1.2 stays. usage: TFB_M3VER=v3 py -B -X utf8 g286_og_temp_logloss.py"""
import os, sys, pickle
os.environ.setdefault("TFB_M3VER", "v3")
import numpy as np, pandas as pd
sys.path.insert(0, r"C:/Users/user/tfb/src")
import t2_m3_gate as G, m3v2
W = "C:/Users/user/tfb/work/t2/"; TS = np.array([0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.8, 2.0]); EPS = 1e-7
lg = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))


def per_window(p, y, msk, pr):
    """k-weighted sums over the scored cells for each T: (log loss, Brier, log loss mid, n cells, n mid cells)"""
    p, y = p[msk].astype(float), y[msk].astype(float); mid = (p >= 0.02) & (p <= 0.98); z = lg(p); out = np.zeros((len(TS), 3))
    for j, T in enumerate(TS):
        q = np.clip(1 / (1 + np.exp(-z / T)), EPS, 1 - EPS); ll = -(y * np.log(q) + (1 - y) * np.log(1 - q))
        out[j] = (ll.sum(), ((q - y) ** 2).sum(), ll[mid].sum())
    return pr * out, pr * np.array([msk.sum(), mid.sum()], float)


def table(entries):
    """entries: list of (y, mask, {k: (pr, p)}) -> per-window arrays S (n, len(TS), 3) and N (n, 2)"""
    S, N = [], []
    for y, msk, ks in entries:
        s_ = np.zeros((len(TS), 3)); n_ = np.zeros(2)
        for pr, p in ks:
            a, b = per_window(p, y, msk, pr); s_ += a; n_ += b
        S.append(s_); N.append(n_)
    return np.array(S), np.array(N)


D4 = pickle.load(open(W + "g284_og_pretemp_dev.pkl", "rb")); WID = sorted(D4)
EA = [(D4[w]["y"].T.astype(bool), D4[w]["m"].T.astype(bool), [(v[0], v[1]) for v in D4[w]["k"].values()]) for w in WID]
EC = [(D4[w]["y"].T.astype(bool), D4[w]["m"].T.astype(bool), [(v[0], v[4]) for v in D4[w]["k"].values()]) for w in WID]
assert all(e[0].shape == e[2][0][1].shape for e in EA)
D5 = pickle.load(open(W + "g285_jf_pretemp.pkl", "rb")); EJ = [(v["y"], v["e"], list(v["k"].values())) for v in D5.values()]
FR = pd.DataFrame([dict(w=w, cond=G.META[w]["cond"], panel=G.META[w]["panel"], day=G.META[w]["day"], hour=(G.META[w]["t"] % 288) // 12) for w in WID])
WT = m3v2.weights(FR, "private")[0]
SA, NA = table(EA); SC, NC = table(EC); SJ, NJ = table(EJ)


def curve(S, N, wt, ix=None):
    ix = np.arange(len(S)) if ix is None else ix; w_ = wt[ix][:, None, None]; tot = (w_ * S[ix]).sum(axis=0); n = (wt[ix][:, None] * N[ix]).sum(axis=0)
    return np.column_stack([tot[:, 0] / n[0], tot[:, 1] / n[0], tot[:, 2] / n[1]])     # (len(TS), 3): log loss, Brier, log loss mid


rng = np.random.default_rng(0); one = np.ones(len(SA)); oneJ = np.ones(len(SJ)); B = 2000
for nm, S, N, wt in (("M3 A (weighted)", SA, NA, WT), ("M3 A (unweighted)", SA, NA, one), ("M3 capB0 (weighted)", SC, NC, WT), ("M3 capB0 (unweighted)", SC, NC, one), ("JF capB0", SJ, NJ, oneJ)):
    c = curve(S, N, wt); print(f"{nm}: T* log loss {TS[c[:, 0].argmin()]}, Brier {TS[c[:, 1].argmin()]}, mid log loss {TS[c[:, 2].argmin()]} | log loss " + " ".join(f"{T}:{v:.5f}" for T, v in zip(TS, c[:, 0])), flush=True)
tA, tC, tJ = [], [], []
for _ in range(B):
    ix = rng.integers(0, len(SA), len(SA)); jx = rng.integers(0, len(SJ), len(SJ))
    tA.append(TS[curve(SA, NA, WT, ix)[:, 0].argmin()]); tC.append(TS[curve(SC, NC, WT, ix)[:, 0].argmin()]); tJ.append(TS[curve(SJ, NJ, oneJ, jx)[:, 0].argmin()])
tA, tC, tJ = map(np.array, (tA, tC, tJ)); dl = tJ - tC; tAd = tA + dl
q = lambda v: (float(np.quantile(v, 0.05)), float(np.quantile(v, 0.95)))
TA0 = TS[curve(SA, NA, WT)[:, 0].argmin()]; DL0 = TS[curve(SJ, NJ, oneJ)[:, 0].argmin()] - TS[curve(SC, NC, WT)[:, 0].argmin()]
iA, iD, iAd = q(tA), q(dl), q(tAd)
side = lambda i: "above" if i[0] > 1.2 else ("below" if i[1] < 1.2 else "contains")
print(f"T*_M3(A) {TA0} 90% [{iA[0]}, {iA[1]}] ({side(iA)} 1.2) | delta {DL0:+.2f} 90% [{iD[0]:+.2f}, {iD[1]:+.2f}] | T*_M3(A) + delta {TA0 + DL0:.2f} 90% [{iAd[0]:.2f}, {iAd[1]:.2f}] ({side(iAd)} 1.2)", flush=True)
ok = side(iA) == side(iAd) and side(iA) != "contains"
cand = min((TA0, TA0 + DL0), key=lambda v: abs(v - 1.2)) if ok else 1.2
print(f"revised line: both intervals exclude 1.2 on the same side: {ok} -> candidate T {round(cand / 0.05) * 0.05:.2f}" + ("" if ok else " (1.2 stays)"), flush=True)
pd.DataFrame(dict(tA=tA, tC=tC, tJ=tJ)).to_csv(W + "g286_boot.csv", index=False)
