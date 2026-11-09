"""Task 2 experiments: CV with options (cell weighting, extra windows)."""
import pickle, sys, numpy as np, lightgbm as lgb
from cachelib import load, T2_PANELS
from t2_windows import score, F
from t2_model import PanelInputs, priors, best_set, FOLDS, PARAMS
from t2_model2 import feats, step_block, predict_window

RNG = np.random.default_rng(5)
NEG_KEEP = 1.0

def design_w(W, PIs, pr, cond, wmode):
    X, y, g, wt = [], [], [], []
    for w in W:
        if w["cond"] != cond: continue
        Xl, names = feats(PIs[w["panel"]], w["t"], cond, pr[w["panel"]])
        lab = w["label"]; el = w["elig"]
        if wmode == "window": ww = 1000.0 / max(el[-1].sum() if cond=="queue_onset" else el.sum(), 1)
        elif wmode == "size": ww = 100.0 / max((lab & el).sum(), 3)
        else: ww = 1.0
        if cond == "queue_onset":
            X.append(Xl); y.append(lab[-1]); g.append(el[-1]); wt.append(np.full(len(Xl), ww))
        else:
            if NEG_KEEP < 1.0:
                iu, idn = names.index("dq_up"), names.index("dq_dn")
                near = (np.minimum(Xl[:, iu], Xl[:, idn]) <= 6.0) | lab.any(axis=0)
                keep = near | (RNG.random(len(Xl)) < NEG_KEEP)
                wk = np.where(near, 1.0, 1.0 / NEG_KEEP)[keep]
            else:
                keep = np.ones(len(Xl), bool); wk = np.ones(len(Xl))
            for s in range(F):
                X.append(step_block(Xl, names, s + 1)[keep]); y.append(lab[s][keep]); g.append(el[s][keep]); wt.append(ww * wk)
    return np.vstack(X), np.concatenate(y).astype(np.float32), np.concatenate(g), np.concatenate(wt)

def cv(W, Wtrain_extra=None, wmode="none", params=None, rounds=(600, 800), tag=""):
    Ps = {p: load(p, "train") for p in T2_PANELS}; PIs = {p: PanelInputs(Ps[p]) for p in T2_PANELS}
    nmap = {p: Ps[p].n for p in T2_PANELS}
    prm = dict(PARAMS); prm.update(params or {})
    oof = [None] * len(W)
    for months in FOLDS:
        pr = priors(W, nmap, set(months))
        tr = [w for w in W if w["date"][:7] not in months]
        if Wtrain_extra: tr = tr + [w for w in Wtrain_extra if w["date"][:7] not in months]
        for cond, nr in zip(("queue_onset", "queue_ongoing"), rounds):
            X, y, g, wt = design_w(tr, PIs, pr, cond, wmode)
            m = lgb.train(prm, lgb.Dataset(X[g], y[g], weight=wt[g]), nr)
            for k, w in enumerate(W):
                if w["cond"] == cond and w["date"][:7] in months:
                    oof[k] = predict_window(m, PIs[w["panel"]], w["t"], cond, pr[w["panel"]])
    preds = [best_set(o.ravel(), w["elig"].ravel()).reshape(o.shape) for o, w in zip(oof, W)]
    s = score(W, preds); print(tag, "S_queue(all,onset,ongoing):", np.round(s, 4), flush=True)
    return oof, s

if __name__ == "__main__":
    W = pickle.load(open(r"C:/Users/user/tfb/work/t2_windows_train.pkl", "rb"))
    for wm in sys.argv[1].split(","):
        cv(W, wmode=wm, tag=f"w={wm}")
