"""G240 (11/03, ledger #203, the strategy's 22:30 request (2)): every arm closed on the masked-history instrument alone (M3 dev as a
whole) read again on M3 dev's top third of history coverage (G233's instrument: nn_test_m3v3/R.npy[:, :12, 2] > the 2/3 quantile of
the arm's condition among dev windows; April composition, m3v2.summary). Saved per-window results only, nothing recomputed. Hold-out
days are dropped before anything is computed (m3v2.is_holdout; dev = onset 100, ongoing 356). January / February are not read here.
usage: TFB_M3VER=v3 py -B -X utf8 g240_clean_reread2.py"""
import os, sys
os.environ.setdefault("TFB_M3VER", "v3")
import numpy as np, pandas as pd
sys.path.insert(0, r"C:/Users/user/tfb/src")
import m3v2
W = "C:/Users/user/tfb/work/t2/"
d_ = "C:/tfb_data/tfb_nn/nn_test_m3v3/"; R_ = np.load(d_ + "R.npy", mmap_mode="r"); o_ = np.load(d_ + "off.npy"); w_ = [str(x) for x in np.load(d_ + "window_id.npy", allow_pickle=True)]
COV = {x: float(np.asarray(R_[o_[i]:o_[i + 1], :12, 2], float).mean()) for i, x in enumerate(w_)}
KEY = pd.read_csv(W + "g108_rectangles.csv")[["w", "cond", "hour"]]          # G120 has no condition / hour columns
ITEMS = [("49", "g108_rectangles.csv", "A", ["O1", "G1", "G2"]), ("50", "g109_end_hazard.csv", "A", ["H1", "H2"]),
         ("58", "g113_onset_site_calib.csv", "A", ["S"]), ("61", "g114_onset_train_prior.csv", "A", ["W1", "W2", "W3"]),
         ("73", "g120_onset_gate.csv", "A", ["ab_67", "ab_ph", "abc_67", "abc_ph"]), ("34", "g104_onset_mbr.csv", "A", ["bs5", "mbr", "bsc"]),
         ("47", "g107_phys_hazard.csv", "base", ["haz"]), ("33", "g103_haz_mix.csv", "A", ["haz10", "haz20", "haz30"]),
         ("31", "g97_x1_onsetv3.csv", "base", ["x1", "x1k4", "l2sp0.1", "l2sp1", "l2sp10"]), ("166", "g170_m3.csv", "base", ["ac"]),
         ("185", "g219_dev.csv", "a10", ["tw0.1", "tw0.3", "la0.1", "la0.3", "xw0.1", "xw0.3", "cn0.35", "cn0.7", "se0.75", "se0.95", "te1.0", "te1.4", "dd2", "dd4"]),
         ("88", "x1_grid/g133_x1_grid.csv", "l2e4_e4", ["l1e4_e2", "l1e4_e4", "l1e4_e8", "l2e4_e2", "l2e4_e8", "l4e4_e2", "l4e4_e4", "l4e4_e8"]),
         ("102/104", "x1_arms/g142.csv", "A", ["hl3", "hl7", "pl"]), ("103", "x1_arms/g142.csv", "A", ["tta3", "tta5"]), ("103b", "x1_arms/g142b.csv", "A", ["tta3", "tta5"])]
n_arm = 0
for nr, f, a, arms in ITEMS:
    D = pd.read_csv(W + f)
    if "cond" not in D:
        D = D.merge(KEY, on="w", how="left"); assert D.cond.notna().all()
    D = D[~m3v2.is_holdout(D.day.to_numpy())].reset_index(drop=True); D["cov"] = D.w.astype(str).map(COV); assert D["cov"].notna().all()
    for cond, d in D.groupby("cond"):
        d = d.reset_index(drop=True); qe = d["cov"].quantile(2 / 3); top = d[d["cov"] > qe].reset_index(drop=True)
        for b in arms:
            if np.allclose(d[b].to_numpy(), d[a].to_numpy()):
                continue
            n_arm += 1; full = m3v2.summary(d, a, b, "private"); s3 = m3v2.summary(top, a, b, "private", n_boot=2000, n_draw=500)
            z = s3["t2_weighted"] / max(s3["se"], 1e-12); nch = int((~np.isclose(top[b], top[a])).sum())
            print(f"#{nr} {b:8s} {cond.split('_')[1]:7s} dev {len(d)}: all {full['t2_weighted']:+.4f} (SE {full['se']:.4f}) | top third {len(top)} ({nch} change) "
                  f"{s3['t2_weighted']:+.4f} (SE {s3['se']:.4f}, z {z:+.2f}) | closure stands: {s3['t2_weighted'] < 0} | Jan/Feb candidate (z >= +1): {z >= 1}", flush=True)
print("arms read", n_arm)
