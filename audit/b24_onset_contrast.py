"""B24 (10/29, ledger #60, strategy 6b): is there a linear direction in the <= T inputs that separates "this site breaks at
T+30" from "the same site at the same clock time on another day did not"? Training-month onset windows (nn_data_afdem):
positive = the head link of each queued run at T+30; controls = the same link in the onset windows of the same panel on
other days of the same weekday type within +-1 h of clock time where that link did not queue at T+30 (another site did).
Score 1: the physics onset ranker's out-of-fold score (t2_phys_onset features and parameters, 9 folds by month).
Score 2: logistic regression on the standardised network inputs of the link (t2_phys_onset COLS), folds by day.
Score 3 (reference): LightGBM on the same inputs, folds by day. Metric: within-pair concordance P(score_pos > score_ctrl).
Line (#60): logistic >= ranker + 0.03 -> candidate feature for the network; else "no linear signal in <= T" -> close."""
import sys, numpy as np, lightgbm as lgb
sys.path.insert(0, r"C:/Users/user/tfb/src")
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from common import NN
import t2_phys_onset as PO
d = f"{NN}/nn_data_afdem"
X = np.load(d + "/X.npy", mmap_mode="r"); Y = np.load(d + "/Y.npy", mmap_mode="r"); E = np.load(d + "/EL.npy", mmap_mode="r")
off = np.load(d + "/off.npy"); cond = np.load(d + "/cond.npy", allow_pickle=True); pan = np.load(d + "/panel.npy", allow_pickle=True)
tt = np.load(d + "/t.npy"); date = np.array([str(x) for x in np.load(d + "/date.npy", allow_pickle=True)])
ks = [k for k in np.flatnonzero(cond == "queue_onset") if (np.asarray(Y[off[k]:off[k + 1], 5]) > 0).any()]
F, RAW, YY, MON, KK = {}, {}, {}, {}, {}
fr, yr, mr, idx = [], [], [], []
for k in ks:
    Xw = np.asarray(X[off[k]:off[k + 1]]); y = np.asarray(Y[off[k]:off[k + 1], 5]) > 0; e = np.asarray(E[off[k]:off[k + 1], 5]) > 0
    f = PO.feats(Xw); RAW[k] = Xw[:, PO.COLS].astype(np.float32); YY[k] = (y, e)
    fr.append(f[e]); yr.append(y[e].astype(int)); mr.append(np.full(e.sum(), date[k][:7])); idx.append(np.column_stack([np.full(e.sum(), k), np.flatnonzero(e)]))
fr, yr, mr, idx = np.vstack(fr), np.concatenate(yr), np.concatenate(mr), np.vstack(idx)
oof = np.zeros(len(yr))
for m in np.unique(mr):                                            # the ranker, out of fold by month
    tr = mr != m; oof[~tr] = lgb.train(PO.PRM, lgb.Dataset(fr[tr], yr[tr]), 400).predict(fr[~tr])
RK = {(int(a), int(b)): s for (a, b), s in zip(idx, oof)}
tod = (tt % 288).astype(int); wk = np.array([np.datetime64(x).astype("datetime64[D]").item().weekday() >= 5 for x in date])
pos, ctrl, groups = [], [], []
for k in ks:
    y, e = YY[k]; dd = np.diff(np.r_[0, y.astype(int), 0]); heads = np.flatnonzero(dd == -1) - 1
    cand = [k2 for k2 in ks if pan[k2] == pan[k] and date[k2] != date[k] and wk[k2] == wk[k] and abs(int(tod[k2]) - int(tod[k])) <= 12]
    for j in heads:
        if not e[j] or (k, j) not in RK:
            continue
        cs = [k2 for k2 in cand if j < len(YY[k2][0]) and YY[k2][1][j] and not YY[k2][0][j] and (k2, j) in RK][:5]
        if cs:
            pos.append((k, j)); ctrl.append(cs); groups.append(date[k])
print("pairs: positives", len(pos), "| controls", sum(len(c) for c in ctrl), flush=True)
# rows for the learned scores: one row per (window, link) in the pairs
rowset = sorted({p for p in pos} | {(k2, j) for (k, j), cs in zip(pos, ctrl) for k2 in cs})
ri = {r: i for i, r in enumerate(rowset)}
Xr = np.vstack([RAW[k][j] for k, j in rowset]); Xr = np.where(np.isfinite(Xr), Xr, np.nan)
lab = np.zeros(len(rowset)); dgrp = np.array([date[k] for k, _ in rowset])
for p in pos:
    lab[ri[p]] = 1
med = np.nan_to_num(np.nanmedian(Xr, axis=0)); Xf = np.where(np.isnan(Xr), med[None, :], Xr)   # all-NaN columns -> 0
days = np.unique(dgrp); fold_of = {dd: i % 5 for i, dd in enumerate(np.random.default_rng(24).permutation(days))}
import os
if os.environ.get("B24_FOLD") == "panel":                       # 10/29: leave-panel-out (no site seen in training)
    pg = np.array([str(pan[k]) for k, _ in rowset]); up = sorted(set(pg)); fold = np.array([up.index(x) % 5 for x in pg])
elif os.environ.get("B24_FOLD") == "month":                     # 10/29: by month (the March / April shift is a month shift)
    mg = np.array([d_[:7] for d_ in dgrp]); um = sorted(set(mg)); fold = np.array([um.index(x) % 5 for x in mg])
else:
    fold = np.array([fold_of[x] for x in dgrp])
s_lr = np.zeros(len(lab)); s_gb = np.zeros(len(lab))
for f in range(5):
    tr = fold != f
    sc = StandardScaler().fit(Xf[tr]); lr = LogisticRegression(max_iter=2000, C=1.0).fit(sc.transform(Xf[tr]), lab[tr])
    s_lr[~tr] = lr.predict_proba(sc.transform(Xf[~tr]))[:, 1]
    s_gb[~tr] = lgb.train(dict(objective="binary", learning_rate=0.05, num_leaves=15, min_data_in_leaf=50, verbose=-1, num_threads=8),
                          lgb.Dataset(Xr[tr], lab[tr]), 300).predict(Xr[~tr])


def conc(score):
    c = []
    for p, cs in zip(pos, ctrl):
        sp = score(p); c += [1.0 if sp > score(q) else (0.5 if sp == score(q) else 0.0) for q in ((k2, p[1]) for k2 in cs)]
    return float(np.mean(c))


r_rank = conc(lambda r: RK[r]); r_lr = conc(lambda r: s_lr[ri[r]]); r_gb = conc(lambda r: s_gb[ri[r]])
print(f"within-pair concordance: ranker (out of fold) {r_rank:.3f} | logistic on the inputs {r_lr:.3f} | LightGBM on the inputs {r_gb:.3f}")
print(f"line: logistic - ranker = {r_lr - r_rank:+.3f} (needs >= +0.03)")
