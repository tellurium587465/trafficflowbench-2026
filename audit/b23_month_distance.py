"""B23 (10/29, ledger #59, strategy 6a): how far is each training month from the test month in the network's <= T inputs?
Adversarial AUC per training month: 300 pseudo-windows of that month (nn_data_afdem, condition mix matched to the test
windows) vs the official windows of March (validation) / April (private) (nn_test_afdem), link rows, the network's
inputs without the columns the production networks zero (pos, tod, dow, capl, lanes, length, vf, w) and the priors;
LightGBM, 5 folds grouped by window, AUC on the rows. Diagnostic only."""
import sys, numpy as np, lightgbm as lgb
sys.path.insert(0, r"C:/Users/user/tfb/src")
from sklearn.metrics import roc_auc_score
from common import NN
d = f"{NN}/nn_data_afdem"; t_ = f"{NN}/nn_test_afdem"
X = np.load(d + "/X.npy", mmap_mode="r"); off = np.load(d + "/off.npy"); cond = np.load(d + "/cond.npy", allow_pickle=True)
date = np.load(d + "/date.npy", allow_pickle=True); names = [str(x) for x in np.load(d + "/names.npy", allow_pickle=True)]
Xt = np.load(t_ + "/X.npy"); offt = np.load(t_ + "/off.npy"); condt = np.load(t_ + "/cond.npy", allow_pickle=True); wid = np.load(t_ + "/window_id.npy", allow_pickle=True)
DROP = {"pos", "tod", "dow", "capl", "lanes", "length", "vf", "w", "prior_on", "prior_og"}
keep = [i for i, n in enumerate(names) if n not in DROP]
mon = np.array([str(x)[:7] for x in date]); rng = np.random.default_rng(23)
prm = dict(objective="binary", learning_rate=0.05, num_leaves=31, min_data_in_leaf=100, feature_fraction=0.8, verbose=-1, num_threads=8)


def rows(Xa, o, idx):
    R, G = [], []
    for g, k in enumerate(idx):
        R.append(np.asarray(Xa[o[k]:o[k + 1]])[:, keep]); G.append(np.full(o[k + 1] - o[k], g))
    return np.vstack(R).astype(np.float32), np.concatenate(G)


for split in ("validation", "private"):
    ti = np.flatnonzero([f"_{split}_" in str(w) for w in wid]); Rt, Gt = rows(Xt, offt, ti)
    share_on = np.mean(condt[ti] == "queue_onset"); out = {}
    for m in sorted(set(mon)):
        on = np.flatnonzero((mon == m) & (cond == "queue_onset")); og = np.flatnonzero((mon == m) & (cond == "queue_ongoing"))
        n_on = min(len(on), int(round(300 * share_on))); pick = np.r_[rng.choice(on, n_on, replace=False), rng.choice(og, 300 - n_on, replace=False)]
        Rm, Gm = rows(X, off, pick)
        Xa = np.vstack([Rm, Rt]); ya = np.r_[np.zeros(len(Rm)), np.ones(len(Rt))]; ga = np.r_[Gm, Gt + Gm.max() + 1]
        fold = rng.integers(0, 5, ga.max() + 1)[ga]; pr = np.zeros(len(ya))
        for f in range(5):
            tr = fold != f; pr[~tr] = lgb.train(prm, lgb.Dataset(Xa[tr], ya[tr]), 200).predict(Xa[~tr])
        out[m] = roc_auc_score(ya, pr)
    a = np.array(list(out.values()))
    print(f"{split}: adversarial AUC by training month " + " ".join(f"{m[2:]} {v:.3f}" for m, v in out.items()) +
          f" | nearest {min(out, key=out.get)} {a.min():.3f}, farthest {max(out, key=out.get)} {a.max():.3f}, range {a.max() - a.min():.3f}", flush=True)
