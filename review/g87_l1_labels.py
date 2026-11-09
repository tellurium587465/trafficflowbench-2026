"""G87 (INSTRUMENT_V3_SPEC §2.3, 10/23): L1 noise-reduced labels for the M3 windows. A cell (row r, link i) is queued under
L1 when the median of the observed speeds at rows r-1, r, r+1 is <= vcut (at least 2 of the 3 observed and eligible);
its mask m1 = at least 2 of the 3 rows observed (and eligible). Labels inside a congestion stay (far from the
threshold); flickering cells at the edge are smoothed. Adds y1 / m1 to every window of m3<ver>_meta.pkl (in place, the
L0 labels y / m untouched). Reports the share of labelled cells whose state differs between L0 and L1.
usage: TFB_M3VER=v3 python g87_l1_labels.py"""
import os, sys, pickle
sys.path.insert(0, r"C:/Users/user/tfb/src")
import numpy as np
from cachelib import load
VER = os.environ.get("TFB_M3VER", ""); W = "C:/Users/user/tfb/work/t2/"
M = pickle.load(open(W + f"m3{VER}_meta.pkl", "rb")); PC = {}; diff = []; cov = []
for m in M["meta"]:
    p = m["panel"]
    if p not in PC:
        P = load(p, "validation"); PC[p] = (P.V, P.vcut, P.E)
    V, vcut, E = PC[p]; t = m["t"]; F = m["y"].shape[1]
    blk = V[t:t + F + 2]                                         # rows t .. t+F+1 -> the 3-row windows of rows t+1 .. t+F
    st = np.stack([blk[0:F], blk[1:F + 1], blk[2:F + 2]])         # (3, F, n)
    nobs = np.isfinite(st).sum(axis=0); med = np.nanmedian(np.where(np.isfinite(st), st, np.nan), axis=0)
    el = E[t + 1:t + F + 1]
    m1 = (nobs >= 2) & el; y1 = m1 & (np.nan_to_num(med, nan=1e9) <= vcut[None, :])
    m["y1"], m["m1"] = y1.T.copy(), m1.T.copy()
    both = m["m"].T.astype(bool) & m1; diff.append(float((m["y"].T.astype(bool)[both] != y1[both]).mean()) if both.any() else 0.0)
    cov.append(float(m1.sum() / max(m["m"].sum(), 1)))
pickle.dump(M, open(W + f"m3{VER}_meta.pkl", "wb"))
print(f"{len(M['meta'])} windows: L1 labels added; cells labelled differently L0 vs L1 {np.mean(diff):.4f}; L1 mask size / L0 {np.mean(cov):.3f}")
