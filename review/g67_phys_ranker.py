"""G67 (10/22 night, rare-site onset reopen, candidate (ii)): how sharp can a LOCATION-AGNOSTIC ranker of the onset link be?
Train months, selector-like onset windows (as G66, cache order = nn_data_afdem). Features: the NN input columns without the
memory / static / clock columns (ZERO list + priors + arr5), each turned into its within-window percentile (so a link is
described only relative to the other links of the same corridor at T), plus the raw values. LightGBM lambdarank, one
group per window, label = queued at T+30 (eligible). Leave-one-month-out over the 9 train months: rank of the best true
link (top-1 / top-3 / top-5) for new-site onsets (leave-month-out rarity <= 0.1, as G66) vs usual sites. Then trained on
all train months and applied to the M3 dev onset windows (hold-out excluded): the same ranks for rare / common truth,
next to the NN's (G64: rare truth top-3 0.62).
usage: python g67_phys_ranker.py"""
import sys, os, pickle
os.environ["TFB_T2CACHE"] = "2_noT_af"
sys.path.insert(0, r"C:/Users/user/tfb/src")
sys.argv = [sys.argv[0], "0"]
import numpy as np, pandas as pd, lightgbm as lgb
from scipy.stats import rankdata
from t2_eval_oof import T
from common import NN
import m3v2

XN = np.load(f"{NN}/nn_data_afdem/X.npy", mmap_mode="r"); OFFN = np.load(f"{NN}/nn_data_afdem/off.npy")
NAMES = [str(x) for x in np.load(f"{NN}/nn_data_afdem/names.npy", allow_pickle=True)]
DROP = {"prior_on", "prior_og", "pos", "vf", "capl", "lanes", "length", "w", "tod", "dow", "arr5", "cond"}
COLS = [i for i, n in enumerate(NAMES) if n not in DROP]
R66 = pd.read_csv("C:/Users/user/tfb/work/t2/g66_rare_context.csv")          # k, month, new (leave-month-out rarity)
NEW = dict(zip(R66.k, R66.new)); MON = dict(zip(R66.k, R66.month))


def feats(Xw):
    X = Xw[:, COLS].astype(float); P = np.full_like(X, 0.5)
    for c in range(X.shape[1]):
        v = X[:, c]; ok = np.isfinite(v)
        if ok.sum() >= 2 and np.nanstd(v) > 0:
            P[ok, c] = (rankdata(v[ok]) - 0.5) / ok.sum()
    return np.column_stack([P, X])


Xs, ys, gs, ks = [], [], [], []
for k in R66.k:
    lab, el = T.LAB[k]; Xw = np.asarray(XN[OFFN[k]:OFFN[k + 1]]); m = el[5]
    Xs.append(feats(Xw)[m]); ys.append(lab[5][m].astype(int)); gs.append(int(m.sum())); ks.append(k)
PRM = dict(objective="lambdarank", learning_rate=0.05, num_leaves=31, min_data_in_leaf=50, feature_fraction=0.8,
           bagging_fraction=0.8, bagging_freq=1, lambdarank_truncation_level=10, verbose=-1, num_threads=8, seed=7)


def ranks(score, y):
    order = np.argsort(-score); r = np.empty(len(score), int); r[order] = np.arange(len(score))
    return int(r[y.astype(bool)].min())


rows = []
for mo in sorted(set(MON.values())):
    tr = [i for i, k in enumerate(ks) if MON[k] != mo]; te = [i for i, k in enumerate(ks) if MON[k] == mo]
    d = lgb.Dataset(np.vstack([Xs[i] for i in tr]), np.concatenate([ys[i] for i in tr]), group=[gs[i] for i in tr])
    mdl = lgb.train(PRM, d, 300)
    for i in te:
        if ys[i].any():
            rows.append(dict(k=ks[i], new=NEW[ks[i]], rank=ranks(mdl.predict(Xs[i]), ys[i]), n=gs[i]))
R = pd.DataFrame(rows)
for name, g in (("new-site", R[R.new]), ("usual", R[~R.new])):
    print(f"train months LOMO, {name}: n {len(g)}, top-1 {np.mean(g['rank'] == 0):.2f}, top-3 {np.mean(g['rank'] < 3):.2f}, "
          f"top-5 {np.mean(g['rank'] < 5):.2f}, median rank {g['rank'].median():.0f} of median {g.n.median():.0f} links")
# all train months -> M3 dev onset windows
mdl = lgb.train(PRM, lgb.Dataset(np.vstack(Xs), np.concatenate(ys), group=gs), 300)
mdl.save_model("C:/Users/user/tfb/work/t2/g67_phys_ranker.txt")
G64 = pd.read_csv("C:/Users/user/tfb/work/t2/g64_onset_rare_diag.csv")
META = {m["window_id"]: m for m in pickle.load(open("C:/Users/user/tfb/work/t2/m3_meta.pkl", "rb"))["meta"]}
XM = np.load(f"{NN}/nn_test_m3/X.npy", mmap_mode="r"); OFFM = np.load(f"{NN}/nn_test_m3/off.npy")
POS = {str(w): i for i, w in enumerate(np.load(f"{NN}/nn_test_m3/window_id.npy", allow_pickle=True))}
out = []
for r in G64.itertuples():
    m = META[r.w]; i = POS[r.w]; Xw = np.asarray(XM[OFFM[i]:OFFM[i + 1]])
    y = m["y"][:, 5].astype(bool) & m["m"][:, 5].astype(bool)
    out.append(dict(w=r.w, rare=r.rare_true, lost=r.iou < 0.2, rank_phys=ranks(mdl.predict(feats(Xw)), y), rank_nn=r.rank_best))
O = pd.DataFrame(out); O.to_csv("C:/Users/user/tfb/work/t2/g67_phys_ranker_m3.csv", index=False)
for name, g in (("rare truth", O[O.rare]), ("common truth", O[~O.rare]), ("lost", O[O.lost])):
    print(f"M3 dev {name}: n {len(g)} | physics top-1 {np.mean(g.rank_phys == 0):.2f} top-3 {np.mean(g.rank_phys < 3):.2f} "
          f"top-5 {np.mean(g.rank_phys < 5):.2f} | NN top-1 {np.mean(g.rank_nn == 0):.2f} top-3 {np.mean(g.rank_nn < 3):.2f} top-5 {np.mean(g.rank_nn < 5):.2f}")
print(O[O.lost].to_string(index=False))
