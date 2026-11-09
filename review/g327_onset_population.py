"""G327 (11/05, ledger #260, measurement only): the population of the networks' training onset windows against the official ones. Training
onset windows = nn_data_afdem's queue_onset windows that are not selector windows (what FINAL trains on); official = the 167 June 2030 selector
onset windows (and, for reference, all selector onset windows of the training months). Per window: the origin's time of day; "first breakdown
of the day" = no queued visible cell on the panel from the day's first row to T - 1 (released train data, unmasked); the site frequency of the
T+30 heads (G198 / #160 f_site, all training months; the smallest over the window's T+30 heads). Line: the first-breakdown share differs by
>= 20 points or the median origin time by >= 2 hours -> a GPU arm is declared separately. usage: py -B -X utf8 g327_onset_population.py"""
import sys, numpy as np, pandas as pd
sys.path.insert(0, r"C:/Users/user/tfb_a8/src")
from cachelib import load, T2_PANELS
NN = "C:/tfb_data/tfb_nn/nn_data_afdem/"; cond = np.load(NN + "cond.npy", allow_pickle=True); isel = np.load(NN + "is_sel.npy"); pan = np.array([str(x) for x in np.load(NN + "panel.npy", allow_pickle=True)])
t_ = np.load(NN + "t.npy"); date = np.load(NN + "date.npy", allow_pickle=True); mon = np.array([str(x)[:7] for x in date]); Y = np.load(NN + "Y.npy", mmap_mode="r"); EL = np.load(NN + "EL.npy", mmap_mode="r"); off = np.load(NN + "off.npy")
BF = r"C:/Users/user/AppData/Local/Temp/claude/C--Users-user-OneDrive-Desktop/1f1db98a-17cd-431b-aafa-61be2b25086b/scratchpad/incident/births_all.parquet"
B = pd.read_parquet(BF); TR = B[(B.split == "train") & (B.nrows >= 2)]; NL = B.groupby("panel").n.first().to_dict(); FS = {}
for p in T2_PANELS:
    T_ = TR[TR.panel == p]; dd = TR.date.unique(); c_ = np.zeros(NL[p])
    for d, g in T_.groupby("date"):
        m = np.zeros(NL[p], bool)
        for h in g.head0.unique():
            m[max(h - 2, 0):h + 3] = True
        c_ += m
    FS[p] = c_ / len(dd)
QD = {}
for p in T2_PANELS:
    P = load(p, "train"); QD[p] = (np.isfinite(P.V) & (P.V <= P.vcut[None, :])).any(axis=1); load.cache_clear()


def describe(idx, lab):
    tod, first, fs = [], [], []
    for k in idx:
        p, t = pan[k], int(t_[k]); tod.append((t % 288) / 12); d0 = t - t % 288; first.append(not QD[p][d0:t].any())
        y = np.asarray(Y[off[k]:off[k + 1]]).astype(bool) & np.asarray(EL[off[k]:off[k + 1]]).astype(bool); yy = y[:, -1] if y.shape[1] == 6 else y[-1]
        hs = [j for j in np.flatnonzero(yy) if j == len(yy) - 1 or not yy[j + 1]]; fs.append(min(FS[p][j] for j in hs) if hs else np.nan)
    tod, first, fs = np.array(tod), np.array(first), np.array(fs)
    print(f"{lab}: {len(idx)} windows | origin time of day median {np.median(tod):.2f} h (IQR {np.quantile(tod, 0.25):.1f}-{np.quantile(tod, 0.75):.1f}) | first breakdown of the day {first.mean():.3f} | head f_site median {np.nanmedian(fs):.3f}, share < 0.02 {np.nanmean(fs < 0.02):.3f}", flush=True)
    return np.median(tod), first.mean()


on = cond == "queue_onset"; tr = np.flatnonzero(on & ~isel); jun = np.flatnonzero(on & isel & (mon == "2030-06")); sel = np.flatnonzero(on & isel)
print("Y shape per window (rows x horizon):", np.asarray(Y[off[tr[0]]:off[tr[0] + 1]]).shape, flush=True)
a = describe(tr, "training onset windows (FINAL trains on these)"); b = describe(jun, "June 2030 official onset windows"); describe(sel, "all training-month official onset windows")
print(f"line (first-breakdown share differs >= 0.20 or median time >= 2 h): {abs(a[1] - b[1]) >= 0.2 or abs(a[0] - b[0]) >= 2}", flush=True)
