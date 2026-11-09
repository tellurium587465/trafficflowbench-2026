"""G251 (11/04, ledger #214, the strategy's (E) from panel 2): E[max] of the final pair on M3 v3 dev (no hold-out): ongoing per window
from G212's dev frame (A = A7, bundle = A8n, a12 = A12 (= A13 on dev: no dense X1 on dev days), B = B7, Bc = B9s, b10 = B10), onset per
window from G170's dev frame (base = the evidence rule factor 0.25 as all the finals, B1.0 = factor 1.0, #120's arm; A12's rank-input
net (#17) is not in it). Candidate "B10on1" = B10 with the onset factor 1.0. Draws of 40 onset + 40 ongoing windows (emax.py), outer
bootstrap 200 x inner 200, against the current pair (A12 = A13, B10); ongoing on all dev windows and on the top third of coverage.
January / February: the A flows need their same-day X1 (not computed), so no pair table there.
usage: TFB_M3VER=v3 py -B -X utf8 g251_pair_emax.py"""
import sys, numpy as np, pandas as pd
sys.path.insert(0, r"C:/Users/user/tfb/review"); from emax import emax_table
W = "C:/Users/user/tfb/work/t2/"; og = pd.read_csv(W + "g212_dev.csv"); on = pd.read_csv(W + "g170_m3.csv"); assert len(og) == 356 and len(on) == 100
d_ = "C:/tfb_data/tfb_nn/nn_test_m3v3/"; R_ = np.load(d_ + "R.npy", mmap_mode="r"); o_ = np.load(d_ + "off.npy"); w_ = [str(x) for x in np.load(d_ + "window_id.npy", allow_pickle=True)]
COV = {x: float(np.asarray(R_[o_[i]:o_[i + 1], :12, 2], float).mean()) for i, x in enumerate(w_)}; og["cov"] = og.w.astype(str).map(COV); top = og[og["cov"] > og["cov"].quantile(2 / 3)].reset_index(drop=True)
C = {"A7": ("base", "A"), "A8n": ("base", "bundle"), "A12": ("base", "a12"), "B7": ("base", "B"), "B9s": ("base", "Bc"), "B10": ("base", "b10"), "B10on1": ("B1.0", "b10")}
P = [(a, b) for a in ("A12", "A8n", "A7") for b in ("B10", "B9s", "B7", "B10on1")] + [("A12", "A8n")]
for lab, g in (("ongoing all dev (356)", og), ("ongoing top coverage third (119)", top)):
    T = emax_table(on, g, C, P, ("A12", "B10")); print(f"--- {lab}; onset all dev (100); E[max] S and the gain over (A12, B10) with its outer-bootstrap SE")
    for p, (lv, gn, se) in sorted(T.items(), key=lambda x: -x[1][1]):
        print(f"  {p[0]:4s} + {p[1]:6s}: E[max] {lv:.5f} | gain {gn:+.5f} (SE {se:.5f}, z {gn / max(se, 1e-12):+.2f})", flush=True)
