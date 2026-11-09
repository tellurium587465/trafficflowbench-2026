"""G163 (11/01, ledger #116 addendum): the M3 - LB level gap, unweighted and raked to the official composition of March (panels equal, weekend x origin hour, m3v2.weights). Final A on the
M3 dev windows (g155 column A_u for onset, g156 column A for ongoing) against the LB split of T2 (#71: onset 0.829, SE 0.049; ongoing 0.881, SE 0.020). Cluster bootstrap (panel x day).
usage: TFB_M3VER=v3 py -B -X utf8 g163_level_gap.py"""
import sys
sys.dont_write_bytecode = True; sys.path.insert(0, r"C:/Users/user/tfb/src"); sys.argv = [sys.argv[0], "0"]
import numpy as np, pandas as pd
import m3v2

W = "C:/Users/user/tfb/work/t2/"; ON = pd.read_csv(W + "g155_onset_b_m3.csv").rename(columns=dict(A_u="A"))[["w", "cond", "panel", "day", "hour", "A"]]; OG = pd.read_csv(W + "g156_m3_frame.csv")[["w", "cond", "panel", "day", "hour", "A"]]
LB = dict(queue_onset=(0.829, 0.049), queue_ongoing=(0.881, 0.020)); rows = {}
for tgt in ("validation", "private"):
    D = pd.concat([ON, OG], ignore_index=True); w, ess = m3v2.weights(D, tgt); cl = pd.factorize(D.panel + "_" + D.day.astype(str))[0]; n = cl.max() + 1; Rm = np.random.default_rng(0).multinomial(n, np.full(n, 1 / n), size=4000); tot = 0; bs = 0
    for c, (lb, se_lb) in LB.items():
        m = (D.cond == c).to_numpy(); a = D.A.to_numpy()[m]; wc = w[m]; cc = cl[m]; lv = float((a * wc).sum() / wc.sum()); b = (Rm @ np.bincount(cc, a * wc, n)) / np.maximum(Rm @ np.bincount(cc, wc, n), 1e-12); se = float(b.std()); tot += lv / 2; bs = bs + b / 2
        print(f"{tgt:10s} {c:14s} unweighted {a.mean():.4f} | raked {lv:.4f} (SE {se:.4f}, ESS {ess[c]:.0f}) | LB {lb} (SE {se_lb}) | LB - raked {lb - lv:+.4f} (SE {np.hypot(se, se_lb):.4f}, z {(lb - lv) / np.hypot(se, se_lb):+.2f}) | LB - unweighted {lb - a.mean():+.4f}")
    print(f"{tgt:10s} T2 raked {tot:.4f} (SE {float(bs.std()):.4f}) | LB 0.855 (SE {np.hypot(.049, .020) / 2:.4f}) | gap {0.855 - tot:+.4f}")
