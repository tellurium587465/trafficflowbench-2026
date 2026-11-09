"""Train Task 2 models on all train windows and predict validation + private windows."""
from __future__ import annotations
import pickle, sys
from pathlib import Path
import numpy as np, pandas as pd, lightgbm as lgb
from cachelib import load, T2_PANELS
from common import REL
from t2_windows import F
from t2_model import PanelInputs, link_features, priors, design, best_set, PARAMS, OUT

ROUNDS = {"queue_onset": 600, "queue_ongoing": 800}


def fit_all(W, PIs, nmap):
    pr = priors(W, nmap, set())
    models = {}
    for cond in ROUNDS:
        X, y, g, _, names = design(W, PIs, pr, cond)
        models[cond] = lgb.train(PARAMS, lgb.Dataset(X[g], y[g]), ROUNDS[cond])
    return models, pr


def predict_split(models, pr, split, select=best_set):
    out = []
    for panel in T2_PANELS:
        P = load(panel, split); PI = PanelInputs(P)
        ti = {s: i for i, s in enumerate(P.times)}
        wi = pd.read_csv(REL / "task2" / panel / split / "window_index.csv")
        for r in wi.itertuples():
            t = ti[r.forecast_origin]
            Xl, _ = link_features(PI, t, r.condition, *pr[panel])
            n = len(Xl)
            if r.condition == "queue_onset":
                p = np.zeros((F, n)); p[-1] = models[r.condition].predict(Xl, num_threads=16)
            else:
                p = np.stack([models[r.condition].predict(np.column_stack([Xl, np.full(n, s + 1)]), num_threads=16) for s in range(F)])
            sel = select(p.ravel(), np.ones(p.size, bool)).reshape(p.shape)
            stamps = [P.times[t + s + 1] for s in range(F)]
            out.append(pd.DataFrame({"window_id": r.window_id, "timestamp": np.repeat(stamps, n),
                                     "link_id": np.tile(P.links, F), "queue_pred": sel.ravel().astype(int)}))
    return pd.concat(out, ignore_index=True)


if __name__ == "__main__":
    tag = sys.argv[1] if len(sys.argv) > 1 else "v1"
    W = pickle.load(open(r"C:/Users/user/tfb/work/t2_windows_train.pkl", "rb"))
    Ps = {p: load(p, "train") for p in T2_PANELS}
    PIs = {p: PanelInputs(Ps[p]) for p in T2_PANELS}
    models, pr = fit_all(W, PIs, {p: Ps[p].n for p in T2_PANELS})
    q = pd.concat([predict_split(models, pr, s) for s in ("validation", "private")], ignore_index=True)
    q.to_parquet(OUT / f"queue_{tag}.parquet", index=False)
    print("rows", len(q), "positives", int(q.queue_pred.sum()))
