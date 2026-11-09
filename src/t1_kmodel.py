"""Direct density (k = q/v) model for Task 3, evaluated on February.

Train a LightGBM on log(k_true) - log(k_base) with weights ~ N = k*L (what the conservation
residual is measured in), then reconcile the (v, q) predictions of the f3big models with k_hat:
log q' = log q + a*(log k_hat - log k), log v' = log v - (1-a)*(log k_hat - log k).
"""
from __future__ import annotations
import sys, numpy as np, lightgbm as lgb
from pathlib import Path
from cachelib import load
from common import PANELS
from t1_exp import featfns
from metrics import state_score, physics_score

MD = Path(r"C:/Users/user/tfb/work/t1")
PANELS3 = sys.argv[1].split(",") if len(sys.argv) > 1 else ["D7_I405_N", "D12_I5_N", "D7_I10_W"]
gf, rf, bf = featfns(3)
mv = lgb.Booster(model_file=str(MD / "speed_f3big.txt")); mq = lgb.Booster(model_file=str(MD / "flow_f3big.txt"))
rng = np.random.default_rng(3)
Xtr, ytr, wtr, cache = [], [], [], {}
for panel in PANELS3:
    pi = PANELS.index(panel); P = load(panel, "train"); F, _, _, _ = gf(P); bv, bq = bf(F)
    r, c = np.nonzero(P.target & P.E & np.isfinite(P.V))
    feb = np.array([P.times[i].startswith("2031-02") for i in r])
    kb = (bq * P.lanes[None, :]) / np.maximum(bv * P.vf[None, :], 1.0)
    hard = np.nan_to_num(F["min_vr_nb"][r, c], nan=1.0) < 0.9
    cand = np.flatnonzero(~feb); pr = np.where(hard[cand], 5.0, 1.0); pr /= pr.sum()
    tr = rng.choice(cand, size=min(400000, len(cand)), replace=False, p=pr)
    X, _ = rf(P, F, r[tr], c[tr]); X = np.column_stack([np.full(len(tr), pi, np.float32), X])
    kt = P.Q[r[tr], c[tr]] / np.maximum(P.V[r[tr], c[tr]], 1.0)
    Xtr.append(X); ytr.append(np.log(kt) - np.log(np.maximum(kb[r[tr], c[tr]], 1e-3)))
    wtr.append(kt * P.length[c[tr]] / (pr[np.searchsorted(cand, tr)] * len(cand)))
    va = np.flatnonzero(feb)
    Xf, _ = rf(P, F, r[va], c[va]); Xf = np.column_stack([np.full(len(va), pi, np.float32), Xf])
    cache[panel] = (r[va], c[va], bv[r[va], c[va]], bq[r[va], c[va]], kb[r[va], c[va]], Xf)
    print(panel, "rows", len(tr), flush=True)
X = np.vstack(Xtr); y = np.concatenate(ytr); w = np.concatenate(wtr); w = w / w.mean()
prm = dict(objective="regression", learning_rate=0.05, num_leaves=255, min_data_in_leaf=100, feature_fraction=0.8,
           bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, num_threads=16, verbose=-1, seed=1)
mk = lgb.train(prm, lgb.Dataset(X, y, weight=w, categorical_feature=[0]), 1500)
mk.save_model(str(MD / "kmodel_3p.txt"))
res = {}
for panel in PANELS3:
    P = load(panel, "train"); r, c, bv, bq, kb, Xf = cache[panel]
    v = (bv + mv.predict(Xf, num_threads=16)) * P.vf[c]; q = (bq + mq.predict(Xf, num_threads=16)) * P.lanes[c]
    khat = np.exp(mk.predict(Xf, num_threads=16)) * np.maximum(kb, 1e-3)
    d = np.log(khat) - np.log(q / np.maximum(v, 1))
    feb = np.array([s.startswith("2031-02") for s in P.times])
    tg, rg = P.target.copy(), P.regime.copy(); P.target = P.target & feb[:, None]; P.regime = np.where(feb, P.regime, 255).astype(np.uint8)
    for a in (None, 1.0, 0.85, 0.7, 0.5):
        if a is None:
            vv, qq = v, q; nm = "raw"
        else:
            vv = v * np.exp(-(1 - a) * d); qq = q * np.exp(a * d); nm = f"a{a}"
        Vp = np.full((P.T, P.n), np.nan); Qp = Vp.copy(); Vp[r, c] = vv; Qp[r, c] = qq
        s = state_score(P, Vp, Qp); ph = physics_score(P, Vp, Qp)
        res.setdefault(nm, []).append((s, ph[2], ph[1]))
    P.target, P.regime = tg, rg
for nm, v in res.items():
    a = np.array(v)
    print(f"{nm:6s} S_state {a[:,0].mean():.5f}  S_phys_proxy {a[:,1].mean():.5f}  S_LWR {a[:,2].mean():.5f}  total-eq {0.35*a[:,0].mean()+0.15*a[:,1].mean():.5f}")
