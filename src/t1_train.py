"""Task 1 v1: physics-feature LightGBM residual model.

Train on official train-split masks (Jun-Jan), select on February, report the
February S_state per panel with the exact scorer.
"""
from __future__ import annotations
import sys, time, json
from pathlib import Path
import numpy as np, lightgbm as lgb
from cachelib import load
from common import PANELS, WORK
from t1_features import grid_features, rows
from metrics import state_score, physics_score, family_mean

OUT = Path(WORK) / "t1"; OUT.mkdir(parents=True, exist_ok=True)
RNG = np.random.default_rng(20261006)
N_TRAIN = 300_000


def base_of(F):
    bv = np.where(np.isfinite(F["lin_vr"]), F["lin_vr"], 1.0)
    bq = np.where(np.isfinite(F["ff_s1_ql"]), F["ff_s1_ql"], F["lin_ql"])
    return bv, bq


def collect():
    Xtr, yv, yq, Xva, meta = [], [], [], [], []
    for pi, panel in enumerate(PANELS):
        t = time.time()
        P = load(panel, "train"); F, V, Q, ok = grid_features(P)
        bv, bq = base_of(F)
        r, c = np.nonzero(P.target & P.E & np.isfinite(P.V))
        feb = np.array([P.times[i].startswith("2031-02") for i in r])
        tr = np.flatnonzero(~feb); va = np.flatnonzero(feb)
        tr = RNG.choice(tr, size=min(N_TRAIN, len(tr)), replace=False)
        for idx, store in ((tr, "tr"), (va, "va")):
            X, names = rows(P, F, r[idx], c[idx])
            X = np.column_stack([np.full(len(idx), pi, np.float32), X])
            lanes = P.lanes[c[idx]]
            tv = P.V[r[idx], c[idx]] / P.vf[c[idx]] - bv[r[idx], c[idx]]
            tq = P.Q[r[idx], c[idx]] / lanes - bq[r[idx], c[idx]]
            if store == "tr":
                Xtr.append(X); yv.append(tv); yq.append(tq)
            else:
                Xva.append(X); meta.append((panel, r[idx], c[idx], tv, tq))
        print(f"{panel}: train rows {len(tr)}, feb rows {len(va)}, {time.time()-t:.1f}s", flush=True)
    return np.vstack(Xtr), np.concatenate(yv), np.concatenate(yq), Xva, meta, ["panel"] + names


def fit(X, y, Xv, yvv, name):
    params = dict(objective="regression", learning_rate=0.05, num_leaves=255, min_data_in_leaf=100,
                  feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
                  num_threads=16, verbose=-1, seed=1)
    ds = lgb.Dataset(X, y, categorical_feature=[0], free_raw_data=False)
    dv = lgb.Dataset(Xv, yvv, reference=ds)
    m = lgb.train(params, ds, 3000, valid_sets=[dv], callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(500)])
    m.save_model(str(OUT / f"{name}.txt"))
    print(name, "best_iter", m.best_iteration, flush=True)
    return m


if __name__ == "__main__":
    N_TRAIN = int(sys.argv[1]) if len(sys.argv) > 1 else 300_000
    t0 = time.time()
    X, yv, yq, Xva, meta, names = collect()
    Xv = np.vstack(Xva); yvv = np.concatenate([m[3] for m in meta]); yqv = np.concatenate([m[4] for m in meta])
    json.dump(names, open(OUT / "features.json", "w"))
    mv = fit(X, yv, Xv, yvv, "speed")
    mq = fit(X, yq, Xv, yqv, "flow")
    # February score per panel with the exact scorer
    res = {}
    off = 0
    for (panel, r, c, tv, tq), Xp in zip(meta, Xva):
        P = load(panel, "train"); F, _, _, _ = grid_features(P); bv, bq = base_of(F)
        pv = mv.predict(Xp, num_threads=16); pq = mq.predict(Xp, num_threads=16)
        Vp = np.full((P.T, P.n), np.nan); Qp = Vp.copy()
        Vp[r, c] = (bv[r, c] + pv) * P.vf[c]; Qp[r, c] = (bq[r, c] + pq) * P.lanes[c]
        Vb = np.full((P.T, P.n), np.nan); Qb = Vb.copy()
        Vb[r, c] = bv[r, c] * P.vf[c]; Qb[r, c] = bq[r, c] * P.lanes[c]
        febrows = np.array([s.startswith("2031-02") for s in P.times])
        sub = type("S", (), {})()
        # restrict scoring to February targets
        tgt = P.target.copy(); P.target = P.target & febrows[:, None]
        s_model, det = state_score(P, Vp, Qp, detail=True); s_base = state_score(P, Vb, Qb)
        ph = physics_score(P, Vp, Qp)
        P.target = tgt
        res[panel] = dict(base=s_base, model=s_model, rmse_v=[d["rmse_v"] for d in det], rmse_q=[d["rmse_q"] for d in det], phys=ph)
        print(f"{panel}: Feb S_state base {s_base:.4f} -> model {s_model:.4f}  rmse_v {np.round([d['rmse_v'] for d in det],2)} rmse_q {np.round([d['rmse_q'] for d in det],1)}  phys(FD,LWRp,S) {np.round(ph,4)}", flush=True)
    print("family mean Feb S_state:", round(family_mean({p: v['model'] for p, v in res.items()}), 5),
          " S_phys proxy:", round(family_mean({p: v['phys'][2] for p, v in res.items()}), 5))
    json.dump({p: {k: (v if not isinstance(v, tuple) else list(v)) for k, v in d.items()} for p, d in res.items()}, open(OUT / "feb_scores.json", "w"), default=float, indent=1)
    print("total time", round(time.time() - t0), "s")
