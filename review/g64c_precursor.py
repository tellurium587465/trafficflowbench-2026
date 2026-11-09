"""G64c: does the true onset link stand out in the inputs at T? For each dev onset window of G64, the within-window
percentile of every input column at the true T+30 link(s) (mean over true links; 0.5 = no signal; ties at 0.5), by rare /
common truth and for the lost windows. Columns sorted by |mean percentile - 0.5| for the rare-truth windows.
usage: python g64c_precursor.py"""
import sys, pickle, numpy as np, pandas as pd
from scipy.stats import rankdata
sys.path.insert(0, r"C:/Users/user/tfb/src")
from common import NN
W = "C:/Users/user/tfb/work/t2/"
D = pd.read_csv(W + "g64_onset_rare_diag.csv")
META = {m["window_id"]: m for m in pickle.load(open(W + "m3_meta.pkl", "rb"))["meta"]}
names = [str(x) for x in np.load(f"{NN}/nn_data_afdem/names.npy", allow_pickle=True)]
X = np.load(f"{NN}/nn_test_m3/X.npy", mmap_mode="r"); off = np.load(f"{NN}/nn_test_m3/off.npy")
pos = {str(w): i for i, w in enumerate(np.load(f"{NN}/nn_test_m3/window_id.npy", allow_pickle=True))}
P = {"rare": [], "common": [], "lost": []}
for r in D.itertuples():
    m = META[r.w]; i = pos[r.w]; Xw = np.asarray(X[off[i]:off[i + 1]], dtype=float)
    y = m["y"][:, 5].astype(bool) & m["m"][:, 5].astype(bool); tl = np.flatnonzero(y)
    pc = np.full(len(names), np.nan)
    for c in range(len(names)):
        v = Xw[:, c]; ok = np.isfinite(v)
        if ok.sum() < 3 or np.nanstd(v) == 0 or not ok[tl].any():
            continue
        rk = (rankdata(v[ok]) - 0.5) / ok.sum(); full = np.full(len(v), np.nan); full[ok] = rk
        pc[c] = np.nanmean(full[tl])
    P["rare" if r.rare_true else "common"].append(pc)
    if r.iou < 0.2:
        P["lost"].append(pc)
S = pd.DataFrame({k: np.nanmean(np.array(v), axis=0) for k, v in P.items()}, index=names)
S["n_rare"] = np.sum(np.isfinite(np.array(P["rare"])), axis=0)
S = S.loc[(S.rare - 0.5).abs().sort_values(ascending=False).index]
print("mean within-window percentile at the true link (rare n=%d, common n=%d, lost n=%d)" % (len(P["rare"]), len(P["common"]), len(P["lost"])))
print(S.head(22).round(3).to_string())
