"""B1: train the blank-cell model on train windows Jun-Dec (early stop on Jan), evaluate on Feb windows vs the linear
interpolation. Windows = featcache2_noT META (is_eval | is_sel), unique (panel, t). Saves models + Feb predictions."""
import os, sys, time, pickle, numpy as np, lightgbm as lgb, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, r"C:/Users/user/tfb/src")
from cachelib import load, T2_PANELS
import t2_demand
from t2_demand import profile
from t2_arrival import _dmin
from t1_gapml import view_train, gap_features, G, PARAMS

OUT = "C:/Users/user/tfb/work/t1/"
Z = np.load("C:/Users/user/tfb/work/t2/featcache2_noT.npz", allow_pickle=True)
m = Z["is_eval"] | Z["is_sel"]
wins = {}
for p, t, c in zip(Z["panel"][m], Z["t"][m], Z["cond"][m]):
    wins.setdefault((str(p), int(t)), str(c))
del Z
rng = np.random.default_rng(11)
TR, VA, TE = [], [], []          # (X, y_v, y_q, w) / Feb: (X, yv, yq, base_v, base_q, tmpl, panel, cond, k)
names = None; t0 = time.time()
for p in T2_PANELS:
    P = load(p, "train"); x = np.concatenate([[0.0], np.cumsum(P.length[:-1])]); dm = _dmin(P, x)
    ws = sorted((t, c) for (pp, t), c in wins.items() if pp == p)
    for t, c in ws:
        if t + G + 31 > P.T:
            continue
        mon = str(P.times[t])[:7]
        V, Q = view_train(P, t)
        X, names, bv, bq, kk, jj = gap_features(P, V, Q, t, c, profile(P, mon), dm, x)
        rr = t + kk
        tv = P.V[rr, jj].astype(np.float64); tq = (P.Q[rr, jj] / P.lanes[jj]).astype(np.float64)
        ok = P.E[rr, jj] & np.isfinite(tv) & np.isfinite(tq) & np.isfinite(bv) & np.isfinite(bq)
        if mon == "2031-02":
            TE.append((X[ok], tv[ok], tq[ok], bv[ok], bq[ok], P.target[rr, jj][ok], np.full(ok.sum(), T2_PANELS.index(p)), np.full(ok.sum(), c == "queue_ongoing"), kk[ok]))
            continue
        f = {nm: X[:, i] for i, nm in enumerate(names)}
        near = (np.minimum.reduce([f["dup_pre"], f["ddn_pre"], f["dup_post"], f["ddn_post"]]) <= 3.0) | (np.nan_to_num(f["v_lin"], nan=1) < 0.85)
        pk = np.where(near, 0.5, 0.08); keep = ok & (rng.random(len(ok)) < pk)
        item = (X[keep], tv[keep] - bv[keep], tq[keep] - bq[keep], 1.0 / pk[keep])
        (VA if mon == "2031-01" else TR).append(item)
    print(p, "windows", len(ws), f"{time.time()-t0:.0f}s", "rows tr", sum(len(a[0]) for a in TR), flush=True)
    load.cache_clear(); t2_demand._PROF.clear()
Xtr = np.vstack([a[0] for a in TR]); Xva = np.vstack([a[0] for a in VA])
res = {}
for ch, yi in (("speed", 1), ("flow", 2)):
    ytr = np.concatenate([a[yi] for a in TR]); wtr = np.concatenate([a[3] for a in TR])
    yva = np.concatenate([a[yi] for a in VA]); wva = np.concatenate([a[3] for a in VA])
    ds = lgb.Dataset(Xtr, ytr, weight=wtr, feature_name=names, free_raw_data=False); dv = lgb.Dataset(Xva, yva, weight=wva, reference=ds)
    mdl = lgb.train(PARAMS, ds, 3000, valid_sets=[dv], callbacks=[lgb.early_stopping(100, verbose=False)])
    mdl.save_model(OUT + f"gapml_{ch}_cv.txt"); res[ch] = mdl.best_iteration
    print(ch, "best_iter", mdl.best_iteration, f"{time.time()-t0:.0f}s", flush=True)
del Xtr, Xva, TR, VA
ms = {ch: lgb.Booster(model_file=OUT + f"gapml_{ch}_cv.txt") for ch in ("speed", "flow")}
Xte = np.vstack([a[0] for a in TE]); tv, tq, bv, bq, tm, pi, og, kk = [np.concatenate([a[i] for a in TE]) for i in range(1, 9)]
pv = bv + ms["speed"].predict(Xte, num_threads=8); pq = bq + ms["flow"].predict(Xte, num_threads=8)
pickle.dump(dict(names=names, best=res, tv=tv, tq=tq, bv=bv, bq=bq, pv=pv, pq=pq, tm=tm, pi=pi, og=og, kk=kk), open(OUT + "gapml_cv_feb.pkl", "wb"))
r = lambda a, b, s: float(np.sqrt(np.mean((a[s] - b[s]) ** 2)))
for nm, s in (("all eligible gap cells", np.ones(len(tv), bool)), ("template targets", tm), ("onset windows", tm & ~og), ("ongoing windows", tm & og),
              ("rows T+5..T+30", tm & (kk <= 6)), ("rows T+35..T+90", tm & (kk > 6))):
    print(f"Feb {nm:24s} n={s.sum():8d}  RMSE v  linear {r(bv, tv, s):6.2f}  gapml {r(pv, tv, s):6.2f} | q/lane linear {r(bq, tq, s):6.1f}  gapml {r(pq, tq, s):6.1f}")
imp = ms["speed"].feature_importance("gain"); imp = imp / imp.sum()
print("top speed features:", [(names[i], round(float(imp[i]), 3)) for i in np.argsort(-imp)[:12]])
