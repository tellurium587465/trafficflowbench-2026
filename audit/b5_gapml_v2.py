"""B5: blank-cell model v2 candidates vs v1 (gapml_*_cv) on the same Feb evaluation as b1.
v1 = eval|sel windows (6.2k Jun-Dec), 127 leaves, lr 0.05, 3000 rounds (capped).
v2 = + a random 12k of the extra windows (sampling rate x0.35), 255 leaves, lr 0.08, early stop on Jan (cap 5000)."""
import sys, time, pickle, numpy as np, lightgbm as lgb, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, r"C:/Users/user/tfb/src")
from cachelib import load, T2_PANELS
import t2_demand
from t2_demand import profile
from t2_arrival import _dmin
from t1_gapml import view_train, gap_features, G, PARAMS

OUT = "C:/Users/user/tfb/work/t1/"
Z = np.load("C:/Users/user/tfb/work/t2/featcache2_noT.npz", allow_pickle=True)
core = Z["is_eval"] | Z["is_sel"]
wins = {}
for p, t, c, k in zip(Z["panel"], Z["t"], Z["cond"], core):
    key = (str(p), int(t))
    if key not in wins or k:
        wins[key] = (str(c), bool(k))
del Z
rng = np.random.default_rng(21)
extra = [k for k, v in wins.items() if not v[1]]
pick = set(map(tuple, np.array(extra, dtype=object)[rng.choice(len(extra), min(12000, len(extra)), replace=False)].tolist()))
TR, VA = [], []; t0 = time.time()
for p in T2_PANELS:
    P = load(p, "train"); x = np.concatenate([[0.0], np.cumsum(P.length[:-1])]); dm = _dmin(P, x)
    for (pp, t), (c, is_core) in sorted(wins.items()):
        if pp != p or t + G + 31 > P.T or (not is_core and (pp, t) not in pick):
            continue
        mon = str(P.times[t])[:7]
        if mon == "2031-02":
            continue
        V, Q = view_train(P, t)
        X, names, bv, bq, kk, jj = gap_features(P, V, Q, t, c, profile(P, mon), dm, x)
        rr = t + kk; tv = P.V[rr, jj].astype(float); tq = (P.Q[rr, jj] / P.lanes[jj]).astype(float)
        ok = P.E[rr, jj] & np.isfinite(tv) & np.isfinite(tq) & np.isfinite(bv) & np.isfinite(bq)
        f = {nm: X[:, i] for i, nm in enumerate(names)}
        near = (np.minimum.reduce([f["dup_pre"], f["ddn_pre"], f["dup_post"], f["ddn_post"]]) <= 3.0) | (np.nan_to_num(f["v_lin"], nan=1) < 0.85)
        pk = np.where(near, 0.5, 0.08) * (1.0 if is_core else 0.35); keep = ok & (rng.random(len(ok)) < pk)
        (VA if mon == "2031-01" else TR).append((X[keep].astype(np.float32), tv[keep] - bv[keep], tq[keep] - bq[keep], 1.0 / pk[keep]))
    print(p, f"{time.time()-t0:.0f}s rows", sum(len(a[0]) for a in TR), flush=True)
    load.cache_clear(); t2_demand._PROF.clear()
Xtr = np.vstack([a[0] for a in TR]); Xva = np.vstack([a[0] for a in VA])
prm = dict(PARAMS, num_leaves=255, learning_rate=0.08, min_data_in_leaf=300)
best = {}
for ch, yi in (("speed", 1), ("flow", 2)):
    ytr = np.concatenate([a[yi] for a in TR]); wtr = np.concatenate([a[3] for a in TR])
    yva = np.concatenate([a[yi] for a in VA]); wva = np.concatenate([a[3] for a in VA])
    ds = lgb.Dataset(Xtr, ytr, weight=wtr, feature_name=names); dv = lgb.Dataset(Xva, yva, weight=wva, reference=ds)
    m = lgb.train(prm, ds, 5000, valid_sets=[dv], callbacks=[lgb.early_stopping(100, verbose=False)])
    m.save_model(OUT + f"gapml2_{ch}_cv.txt"); best[ch] = m.best_iteration
    print(ch, "best_iter", m.best_iteration, f"{time.time()-t0:.0f}s", flush=True)
del Xtr, Xva, TR, VA
# Feb evaluation on the same cells as b1 (gapml_cv_feb.pkl keeps the truth/base; rebuild X for Feb windows)
d1 = pickle.load(open(OUT + "gapml_cv_feb.pkl", "rb"))
m2 = {ch: lgb.Booster(model_file=OUT + f"gapml2_{ch}_cv.txt") for ch in ("speed", "flow")}
pv2, pq2 = [], []
for p in T2_PANELS:
    P = load(p, "train"); x = np.concatenate([[0.0], np.cumsum(P.length[:-1])]); dm = _dmin(P, x)
    for (pp, t), (c, is_core) in sorted(wins.items()):
        if pp != p or not is_core or t + G + 31 > P.T or str(P.times[t])[:7] != "2031-02":
            continue
        V, Q = view_train(P, t)
        X, names, bv, bq, kk, jj = gap_features(P, V, Q, t, c, profile(P, "2031-02"), dm, x)
        rr = t + kk; tv = P.V[rr, jj].astype(float); tq = (P.Q[rr, jj] / P.lanes[jj]).astype(float)
        ok = P.E[rr, jj] & np.isfinite(tv) & np.isfinite(tq) & np.isfinite(bv) & np.isfinite(bq)
        pv2.append(bv[ok] + m2["speed"].predict(X[ok], num_threads=8)); pq2.append(bq[ok] + m2["flow"].predict(X[ok], num_threads=8))
    load.cache_clear(); t2_demand._PROF.clear()
pv2, pq2 = np.concatenate(pv2), np.concatenate(pq2)
assert len(pv2) == len(d1["tv"]), (len(pv2), len(d1["tv"]))
r = lambda a, b, s: float(np.sqrt(np.mean((a[s] - b[s]) ** 2)))
for nm, s in (("all eligible gap cells", np.ones(len(pv2), bool)), ("template targets", d1["tm"])):
    print(f"Feb {nm:22s} RMSE v linear {r(d1['bv'], d1['tv'], s):.2f} v1 {r(d1['pv'], d1['tv'], s):.2f} v2 {r(pv2, d1['tv'], s):.2f} avg {r((pv2 + d1['pv']) / 2, d1['tv'], s):.2f} | "
          f"q/lane linear {r(d1['bq'], d1['tq'], s):.1f} v1 {r(d1['pq'], d1['tq'], s):.1f} v2 {r(pq2, d1['tq'], s):.1f} avg {r((pq2 + d1['pq']) / 2, d1['tq'], s):.1f}")
pickle.dump(dict(pv2=pv2, pq2=pq2, best=best), open(OUT + "gapml2_cv_feb.pkl", "wb"))
