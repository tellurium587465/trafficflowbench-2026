"""T1 rounds-cap check (R3 ledger: f3big speed best_iter 5939 of a 6000 cap). Same recipe as t1_big (f3 features,
congestion boost x10, lr 0.05) on 3 panels / 500k rows, cap 12000 with patience 300; Feb S_state at min(6000, best)
vs at best."""
import sys, time, json
sys.path.insert(0, r"C:/Users/user/tfb/src")
import numpy as np, lightgbm as lgb
from cachelib import load
from common import PANELS
from metrics import state_score, physics_score
from t1_exp import featfns, OUT
def run(fv, ntrain, panels, tag, params=None, rounds=4000, save=False, cong_boost=0.0, cut=6000):
    gf, rf, bf = featfns(fv)
    rng = np.random.default_rng(1)
    Xtr, yv, yq, Xes, ev, eq, cache, WT = [], [], [], [], [], [], {}, []
    t0 = time.time()
    for panel in panels:
        pi = PANELS.index(panel)
        P = load(panel, "train"); F, _, _, _ = gf(P); bv, bq = bf(F)
        r, c = np.nonzero(P.target & P.E & np.isfinite(P.V))
        feb = np.array([P.times[i].startswith("2031-02") for i in r])
        cand = np.flatnonzero(~feb)
        if cong_boost > 0:
            hard = np.nan_to_num(F["min_vr_nb"][r[cand], c[cand]], nan=1.0) < 0.9
            pr_ = np.where(hard, cong_boost, 1.0); pr_ = pr_ / pr_.sum()
            tr = rng.choice(cand, size=min(ntrain, len(cand)), replace=False, p=pr_)
            wts = 1.0 / (pr_[np.searchsorted(cand, tr)] * len(cand))
        else:
            tr = rng.choice(cand, size=min(ntrain, len(cand)), replace=False); wts = np.ones(len(tr))
        va = np.flatnonzero(feb)
        es = rng.choice(va, size=min(40000, len(va)), replace=False)
        for idx, which in ((tr, "tr"), (es, "es")):
            X, names = rf(P, F, r[idx], c[idx]); X = np.column_stack([np.full(len(idx), pi, np.float32), X])
            tv = P.V[r[idx], c[idx]] / P.vf[c[idx]] - bv[r[idx], c[idx]]
            tq = P.Q[r[idx], c[idx]] / P.lanes[c[idx]] - bq[r[idx], c[idx]]
            if which == "tr":
                Xtr.append(X); yv.append(tv); yq.append(tq); WT.append(wts)
            else:
                Xes.append(X); ev.append(tv); eq.append(tq)
        Xf, _ = rf(P, F, r[va], c[va]); Xf = np.column_stack([np.full(len(va), pi, np.float32), Xf])
        cache[panel] = (r[va], c[va], bv[r[va], c[va]], bq[r[va], c[va]], Xf)
    X = np.vstack(Xtr); Xe = np.vstack(Xes)
    print(f"[{tag}] features {X.shape} built in {time.time()-t0:.0f}s", flush=True)
    prm = dict(objective="regression", learning_rate=0.08, num_leaves=255, min_data_in_leaf=100,
               feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, num_threads=16, verbose=-1, seed=1, force_row_wise=True)
    if params:
        prm.update(params)
    models = {}
    for nm, y, ye in (("speed", np.concatenate(yv), np.concatenate(ev)), ("flow", np.concatenate(yq), np.concatenate(eq))):
        ds = lgb.Dataset(X, y, weight=np.concatenate(WT), categorical_feature=[0]); de = lgb.Dataset(Xe, ye, reference=ds)
        m = lgb.train(prm, ds, rounds, valid_sets=[de], callbacks=[lgb.early_stopping(300, verbose=False)])
        models[nm] = m
        if save:
            m.save_model(str(OUT / f"{nm}_{tag}.txt"))
        print(f"[{tag}] {nm} best_iter {m.best_iteration} ({time.time()-t0:.0f}s)", flush=True)
    out = {}
    for NI in ("cut", "best"):
      res = {}
      for panel in panels:
          P = load(panel, "train"); r, c, bv, bq, Xf = cache[panel]
          Vp = np.full((P.T, P.n), np.nan); Qp = Vp.copy()
          ni = lambda m: min(cut, m.best_iteration) if NI == "cut" else m.best_iteration
          Vp[r, c] = (bv + models["speed"].predict(Xf, num_iteration=ni(models["speed"]), num_threads=16)) * P.vf[c]
          Qp[r, c] = (bq + models["flow"].predict(Xf, num_iteration=ni(models["flow"]), num_threads=16)) * P.lanes[c]
          feb = np.array([s.startswith("2031-02") for s in P.times])
          tg, rg = P.target.copy(), P.regime.copy()
          P.target = P.target & feb[:, None]; P.regime = np.where(feb, P.regime, 255).astype(np.uint8)
          s, det = state_score(P, Vp, Qp, detail=True); ph = physics_score(P, Vp, Qp)
          P.target, P.regime = tg, rg
          res[panel] = dict(S=s, phys=ph, rmse_v=[d["rmse_v"] for d in det], rmse_q=[d["rmse_q"] for d in det])
          print(f"[{tag}] {panel}: S_state {s:.4f} rmse_v {np.round(res[panel]['rmse_v'],2)} rmse_q {np.round(res[panel]['rmse_q'],1)} phys {np.round(ph,4)}", flush=True)
      ms = np.mean([v['S'] for v in res.values()]); mp = np.mean([v['phys'][2] for v in res.values()])
      print(f'[{tag}] {NI}: MEAN S_state {ms:.5f} S_phys_proxy {mp:.5f}', flush=True); out[NI] = (ms, mp)
    return out
if __name__ == "__main__":
    print(run(3, 500000, ["D7_I405_N", "D12_I5_N", "D7_I10_W"], "g14", params=dict(learning_rate=0.05), rounds=12000, cong_boost=10.0))
