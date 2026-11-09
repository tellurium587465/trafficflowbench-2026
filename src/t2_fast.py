"""Fast Task 2 CV on the feature cache."""
from __future__ import annotations
import pickle, sys, time, json
import numpy as np, lightgbm as lgb
from cachelib import T2_PANELS, load
from t2_windows import score, F
from t2_model import best_set, FOLDS, PARAMS

import os
_C = os.environ.get("TFB_T2CACHE", "")
Z = np.load(rf"C:/Users/user/tfb/work/t2/featcache{_C}.npz", allow_pickle=True)
X_ALL, OFF, NAMES = Z["X"], Z["off"], list(Z["names"])
META = {k: Z[k] for k in ("panel", "t", "cond", "date", "is_eval")}
META["is_sel"] = Z["is_sel"] if "is_sel" in Z.files else np.zeros(len(Z["off"]) - 1, bool)
LAB = pickle.load(open(rf"C:/Users/user/tfb/work/t2/labels_all{_C}.pkl", "rb"))
NWIN = len(OFF) - 1
I_PON, I_POG = NAMES.index("prior_on"), NAMES.index("prior_og")
I_TAIL, I_SF, I_SE = NAMES.index("seg_tail"), NAMES.index("s_flow"), NAMES.index("s_emp_tail")
I_DU, I_DD = NAMES.index("dq_up"), NAMES.index("dq_dn")
NMAP = {p: load(p, "train").n for p in T2_PANELS}


def priors(exclude_months):
    pr = {p: [np.zeros(n), np.zeros(n), 0, 0] for p, n in NMAP.items()}
    for k in range(NWIN):
        if not META["is_eval"][k] or META["date"][k][:7] in exclude_months:
            continue
        lab, _ = LAB[k]; a = pr[META["panel"][k]]
        if META["cond"][k] == "queue_onset":
            a[0] += lab[-1]; a[2] += 1
        else:
            a[1] += lab.any(axis=0); a[3] += 1
    return {p: (a[0] / max(a[2], 1), a[1] / max(a[3], 1)) for p, a in pr.items()}


DROP = [NAMES.index(c) for c in os.environ.get("TFB_T2_DROP", "").split(",") if c]


def apply_drop(X):
    if DROP:
        X[:, DROP] = np.nan
    return X


USE_MP = os.environ.get("TFB_T2_MP", "0") == "1"
MP_ALL = np.load(rf"C:/Users/user/tfb/work/t2/mpcache{_C}.npy") if USE_MP else None
USE_ARR = os.environ.get("TFB_T2_ARR", "0") == "1"
ARR_V = os.environ.get("TFB_T2_ARRV", "")
ARR_ALL = np.load(rf"C:/Users/user/tfb/work/t2/arrcache{_C}{ARR_V}.npy") if USE_ARR else None
USE_REL = os.environ.get("TFB_T2_REL", "0") == "1"             # corridor-relative hazard (t2_relcache.py)
REL_ALL = np.load(rf"C:/Users/user/tfb/work/t2/relcache{_C}.npy", mmap_mode="r") if USE_REL else None
USE_DEM = os.environ.get("TFB_T2_DEM", "0") == "1"             # weekday x ToD demand forecast (t2_demcache.py)
DEM_ALL = np.load(rf"C:/Users/user/tfb/work/t2/demcache{_C}.npy", mmap_mode="r") if USE_DEM else None


def win_X(k, pr):
    X = X_ALL[OFF[k]:OFF[k + 1]].copy()
    if USE_MP:
        X = np.column_stack([X, MP_ALL[OFF[k]:OFF[k + 1]]])
    if USE_ARR:
        X = np.column_stack([X, ARR_ALL[OFF[k]:OFF[k + 1]]])
    if USE_REL:
        X = np.column_stack([X, REL_ALL[OFF[k]:OFF[k + 1]]])
    if USE_DEM:
        X = np.column_stack([X, DEM_ALL[OFF[k]:OFF[k + 1]]])
    po, pg = pr[META["panel"][k]]
    X[:, I_PON] = po; X[:, I_POG] = pg
    return apply_drop(X)


def step_block(X, s):
    h = s * 5 / 60.0
    return np.column_stack([X, np.full(len(X), s, np.float32), X[:, I_TAIL] + np.nan_to_num(X[:, I_SF]) * h,
                            X[:, I_TAIL] + np.nan_to_num(X[:, I_SE]) * h]).astype(np.float32)


PRIOR_LOMO = False
WPOW = float(os.environ.get("TFB_T2_WPOW", "1"))   # window weight = 100 / max(queued cells, 3) ** WPOW


def design(idx, pr, cond, wmode="size", neg_keep=0.15, rng=None, exclude=()):
    rng = rng or np.random.default_rng(5)
    Xs, ys, ws = [], [], []
    lomo = {}
    for k in idx:
        if META["cond"][k] != cond:
            continue
        if PRIOR_LOMO:
            mth = META["date"][k][:7]
            if mth not in lomo:
                lomo[mth] = priors(set(exclude) | {mth})
            X = win_X(k, lomo[mth])
        else:
            X = win_X(k, pr)
        lab, el = LAB[k]
        ww = 100.0 / max((lab & el).sum(), 3) ** WPOW if wmode == "size" else 1.0
        if cond == "queue_onset":
            m = el[-1]; Xs.append(X[m]); ys.append(lab[-1][m]); ws.append(np.full(m.sum(), ww))
        else:
            near = (np.minimum(X[:, I_DU], X[:, I_DD]) <= 6.0) | lab.any(axis=0)
            keep = near | (rng.random(len(X)) < neg_keep)
            wk = np.where(near, 1.0, 1.0 / neg_keep)
            for s in range(F):
                m = keep & el[s]
                Xs.append(step_block(X, s + 1)[m]); ys.append(lab[s][m]); ws.append(ww * wk[m])
    return np.vstack(Xs), np.concatenate(ys).astype(np.float32), np.concatenate(ws)


def predict(m, k, pr):
    X = win_X(k, pr)
    if META["cond"][k] == "queue_onset":
        p = np.zeros((F, len(X))); p[-1] = m.predict(X, num_threads=16)
    else:
        p = m.predict(np.vstack([step_block(X, s + 1) for s in range(F)]), num_threads=16).reshape(F, len(X))
    return p


def cv(params=None, rounds=(600, 800), wmode="size", neg_keep=0.15, use_extra=True, tag="", save=False, eval_key="is_eval"):
    prm = dict(PARAMS); prm.update(params or {})
    if os.environ.get("TFB_T2_FAST", "0") == "1":                 # screening: ~3-4x faster, same ranking of features
        prm.update(bagging_fraction=0.5, bagging_freq=1, feature_fraction=0.8, max_bin=63)
        rounds = tuple(max(100, r // 2) for r in rounds)
    t0 = time.time()
    eval_idx = np.flatnonzero(META["is_eval"] | META["is_sel"])
    oof = {}
    folds, grp = FOLDS, np.array([d[:7] for d in META["date"]])
    if os.environ.get("TFB_T2_CV", "month") == "panel":          # leave-panels-out: location memory cannot help
        assert I_PON in DROP and I_POG in DROP, "panel CV needs prior_on/prior_og dropped (they would leak)"
        ps = sorted(set(META["panel"])); folds = [tuple(ps[i::4]) for i in range(4)]; grp = META["panel"]
    for months in folds:
        pr = priors(set(months))
        tr = np.flatnonzero(~np.isin(grp, months) & (META["is_eval"] | use_extra) & ~META["is_sel"])
        te = [k for k in eval_idx if grp[k] in months]
        for cond, nr in zip(("queue_onset", "queue_ongoing"), rounds):
            X, y, w = design(tr, pr, cond, wmode, neg_keep, exclude=months)
            m = lgb.train(prm, lgb.Dataset(X, y, weight=w), nr)
            for k in te:
                if META["cond"][k] == cond:
                    oof[k] = predict(m, k, pr)
    res = {}
    for key in ("is_eval", "is_sel"):
        idx = [k for k in eval_idx if META[key][k]]
        if not idx:
            continue
        Wv = [dict(panel=META["panel"][k], cond=META["cond"][k], label=LAB[k][0], elig=LAB[k][1]) for k in idx]
        preds = [best_set(oof[k].ravel(), LAB[k][1].ravel()).reshape(oof[k].shape) for k in idx]
        res[key] = score(Wv, preds)
        print(f"{tag} [{key}] S_queue(all,onset,ongoing): {np.round(res[key],4)}  ({time.time()-t0:.0f}s)", flush=True)
    s = res.get(eval_key)
    if save:
        pickle.dump({int(k): oof[k] for k in eval_idx}, open(rf"C:/Users/user/tfb/work/t2/oof_{tag}.pkl", "wb"))
    return s, oof


if __name__ == "__main__":
    cv(tag="base", save=True)
