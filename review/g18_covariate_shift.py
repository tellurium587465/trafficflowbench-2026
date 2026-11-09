"""G18: covariate shift of the NN inputs between the train windows and the test windows (inputs only, no labels;
the adversarial-validation idea). Per feature, KS distance between train-window link rows (selector-like windows,
by condition) and test-window link rows (validation, private). Compared with the KS spread between train month folds
(the shift the local yardstick already sees). Features with test KS far above the fold KS are transport risks."""
import sys, numpy as np, pandas as pd
from scipy.stats import ks_2samp
D = "E:/tfb_nn/nn_data_afdem"; DT = "E:/tfb_nn/nn_test_afdem"
names = list(np.load(f"{D}/names.npy", allow_pickle=True))
X = np.load(f"{D}/X.npy", mmap_mode="r"); off = np.load(f"{D}/off.npy"); cond = np.load(f"{D}/cond.npy", allow_pickle=True)
date = np.load(f"{D}/date.npy", allow_pickle=True); sel = np.load(f"{D}/is_sel.npy")
Xt = np.load(f"{DT}/X.npy", mmap_mode="r"); offt = np.load(f"{DT}/off.npy"); condt = np.load(f"{DT}/cond.npy", allow_pickle=True)
wid = np.load(f"{DT}/window_id.npy", allow_pickle=True)
rng = np.random.default_rng(0)


def rows(Xa, o, idx, cap=60000):
    r = np.concatenate([np.arange(o[i], o[i + 1]) for i in idx])
    if len(r) > cap:
        r = np.sort(rng.choice(r, cap, replace=False))
    return np.asarray(Xa[r])


mon = np.array([str(d)[:7] for d in date])
FOLD = {**{m: 0 for m in ("2030-06", "2030-07", "2030-08")}, **{m: 1 for m in ("2030-09", "2030-10", "2030-11")},
        **{m: 2 for m in ("2030-12", "2031-01", "2031-02")}}
fold = np.array([FOLD.get(m, -1) for m in mon])
sp = np.array(["validation" if "validation" in str(w) else "private" for w in wid])
skip = {"pos", "tod", "dow", "capl", "lanes", "length", "vf", "w", "prior_on", "prior_og", "cond"}
for c in ("queue_onset", "queue_ongoing"):
    tr = np.flatnonzero(sel & (cond == c)); A = rows(X, off, tr)
    Fd = [rows(X, off, np.flatnonzero(sel & (cond == c) & (fold == f)), 20000) for f in range(3)]
    Tv = rows(Xt, offt, np.flatnonzero((condt == c) & (sp == "validation"))); Tp = rows(Xt, offt, np.flatnonzero((condt == c) & (sp == "private")))
    out = []
    for j, n in enumerate(names):
        if n in skip:
            continue
        def ks(a, b):
            a, b = a[np.isfinite(a)], b[np.isfinite(b)]
            return ks_2samp(a, b).statistic if len(a) > 50 and len(b) > 50 else np.nan
        fk = np.nanmax([ks(Fd[f][:, j], np.concatenate([Fd[g][:, j] for g in range(3) if g != f])) for f in range(3)])
        out.append((n, fk, ks(A[:, j], Tv[:, j]), ks(A[:, j], Tp[:, j]), np.nanmedian(A[:, j]), np.nanmedian(Tv[:, j]), np.nanmedian(Tp[:, j])))
    R = pd.DataFrame(out, columns=["feature", "ks_fold_max", "ks_val", "ks_priv", "med_train", "med_val", "med_priv"])
    R["excess"] = R[["ks_val", "ks_priv"]].max(axis=1) - R.ks_fold_max
    print(f"== {c}: train rows {len(A)}, val {len(Tv)}, private {len(Tp)}")
    print(R.sort_values("excess", ascending=False).head(14).round(3).to_string(index=False))
