"""G72 (strategy §17, 10/22): are the train-month 'new-site' onsets (leave-month-out rarity, G66, 69 windows) the same
kind as M3's rare-site onsets? Within-window percentile of the physics columns at the true T+30 link (as G64c) for the
train-month new-site and usual windows; compare with M3 (G64c: rare truth dem30_over_dnmin 0.88, arr6 0.87, nb2_q 0.83).
usage: python g72_rare_population.py"""
import sys, os
os.environ["TFB_T2CACHE"] = "2_noT_af"
sys.path.insert(0, r"C:/Users/user/tfb/src")
sys.argv = [sys.argv[0], "0"]
import numpy as np, pandas as pd
from scipy.stats import rankdata
from t2_eval_oof import T
from common import NN
XN = np.load(f"{NN}/nn_data_afdem/X.npy", mmap_mode="r"); OFFN = np.load(f"{NN}/nn_data_afdem/off.npy")
NAMES = [str(x) for x in np.load(f"{NN}/nn_data_afdem/names.npy", allow_pickle=True)]
COLS = ["dem30_over_dnmin", "arr6", "nb2_q", "nb2_qmax", "nb1_q", "q_trend", "dem30_c", "q_today"]
R66 = pd.read_csv("C:/Users/user/tfb/work/t2/g66_rare_context.csv")
out = {True: [], False: []}
for r in R66.itertuples():
    lab, el = T.LAB[r.k]; y = (lab[5] & el[5]).astype(bool); tl = np.flatnonzero(y)
    Xw = np.asarray(XN[OFFN[r.k]:OFFN[r.k + 1]], dtype=float); row = []
    for c in COLS:
        v = Xw[:, NAMES.index(c)]; ok = np.isfinite(v)
        if ok.sum() < 3 or np.std(v[ok]) == 0 or not ok[tl].any():
            row.append(np.nan); continue
        full = np.full(len(v), np.nan); full[ok] = (rankdata(v[ok]) - 0.5) / ok.sum(); row.append(np.nanmean(full[tl]))
    out[bool(r.new)].append(row)
S = pd.DataFrame({"train new-site (n=%d)" % len(out[True]): np.nanmean(out[True], axis=0),
                  "train usual (n=%d)" % len(out[False]): np.nanmean(out[False], axis=0)}, index=COLS)
S["M3 rare (G64c)"] = [0.877, 0.871, 0.832, 0.825, 0.793, 0.693, 0.679, 0.726]
S["M3 common (G64c)"] = [0.838, 0.853, 0.704, 0.699, 0.713, 0.611, 0.635, 0.906]
print(S.round(3).to_string())
