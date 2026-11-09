"""B3: is the blank-cell model location-robust?  Hold out whole panels (never seen in training), train on the other
panels Jun-Dec (early stop on their Jan), test on the held-out panels' Feb windows.  Variants: all features vs
without time-of-day / weekday / static link attributes (the features that could act as a link or panel identity)."""
import sys, time, numpy as np, lightgbm as lgb, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, r"C:/Users/user/tfb/src")
from cachelib import load, T2_PANELS
import t2_demand
from t2_demand import profile
from t2_arrival import _dmin
from t1_gapml import view_train, gap_features, G, PARAMS

HOLD = sys.argv[1].split(",") if len(sys.argv) > 1 else ["D7_I405_N", "D12_I5_S"]
Z = np.load("C:/Users/user/tfb/work/t2/featcache2_noT.npz", allow_pickle=True)
m = Z["is_eval"] | Z["is_sel"]
wins = {}
for p, t, c in zip(Z["panel"][m], Z["t"][m], Z["cond"][m]):
    wins.setdefault((str(p), int(t)), str(c))
del Z
rng = np.random.default_rng(12); TR, VA, TE = [], [], []; names = None; t0 = time.time()
for p in T2_PANELS:
    P = load(p, "train"); x = np.concatenate([[0.0], np.cumsum(P.length[:-1])]); dm = _dmin(P, x)
    for t, c in sorted((t, c) for (pp, t), c in wins.items() if pp == p):
        if t + G + 31 > P.T:
            continue
        mon = str(P.times[t])[:7]
        if (p in HOLD) != (mon == "2031-02"):
            continue
        V, Q = view_train(P, t)
        X, names, bv, bq, kk, jj = gap_features(P, V, Q, t, c, profile(P, mon), dm, x)
        rr = t + kk; tv = P.V[rr, jj].astype(float); tq = (P.Q[rr, jj] / P.lanes[jj]).astype(float)
        ok = P.E[rr, jj] & np.isfinite(tv) & np.isfinite(tq) & np.isfinite(bv) & np.isfinite(bq)
        if p in HOLD:
            ok &= P.target[rr, jj]
            TE.append((X[ok], tv[ok], tq[ok], bv[ok], bq[ok])); continue
        f = {nm: X[:, i] for i, nm in enumerate(names)}
        near = (np.minimum.reduce([f["dup_pre"], f["ddn_pre"], f["dup_post"], f["ddn_post"]]) <= 3.0) | (np.nan_to_num(f["v_lin"], nan=1) < 0.85)
        pk = np.where(near, 0.5, 0.08); keep = ok & (rng.random(len(ok)) < pk)
        (VA if mon == "2031-01" else TR).append((X[keep], tv[keep] - bv[keep], tq[keep] - bq[keep], 1.0 / pk[keep]))
    load.cache_clear(); t2_demand._PROF.clear()
print("rows", sum(len(a[0]) for a in TR), sum(len(a[0]) for a in VA), sum(len(a[0]) for a in TE), f"{time.time()-t0:.0f}s", flush=True)
Xtr = np.vstack([a[0] for a in TR]); Xva = np.vstack([a[0] for a in VA]); Xte = np.vstack([a[0] for a in TE])
tv, tq, bv, bq = [np.concatenate([a[i] for a in TE]) for i in range(1, 5)]
r = lambda a, b: float(np.sqrt(np.mean((a - b) ** 2)))
print(f"held-out {HOLD} Feb template targets n={len(tv)}: linear RMSE v {r(bv, tv):.2f} q/lane {r(bq, tq):.1f}", flush=True)
DROPS = {"all features": [], "no tod/dow/static": ["tod", "dow", "vf", "capl", "lanes", "length", "w", "dmin_rel"]}
prm = dict(PARAMS, learning_rate=0.1)
for nm, drop in DROPS.items():
    keep = [i for i, c in enumerate(names) if c not in drop]
    out = []
    for yi in (1, 2):
        ytr = np.concatenate([a[yi] for a in TR]); wtr = np.concatenate([a[3] for a in TR])
        yva = np.concatenate([a[yi] for a in VA]); wva = np.concatenate([a[3] for a in VA])
        ds = lgb.Dataset(Xtr[:, keep], ytr, weight=wtr); dv = lgb.Dataset(Xva[:, keep], yva, weight=wva, reference=ds)
        mdl = lgb.train(prm, ds, 4000, valid_sets=[dv], callbacks=[lgb.early_stopping(100, verbose=False)])
        out.append((mdl.best_iteration, mdl.predict(Xte[:, keep], num_threads=8)))
    print(f"  {nm:20s} gapml RMSE v {r(bv + out[0][1], tv):.2f} (iter {out[0][0]})  q/lane {r(bq + out[1][1], tq):.1f} (iter {out[1][0]})  {time.time()-t0:.0f}s", flush=True)
