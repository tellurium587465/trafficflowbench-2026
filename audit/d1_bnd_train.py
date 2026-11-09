"""D1: training rows for the boundary specialist from the train months (Task 1 targets of the train masked layer),
month slices Jun-Jan (Feb stays held out for the simulation), all 10 panels.  Train on Jun-Dec, early stop on Jan,
then refit on Jun-Jan with the best rounds.  Saves work/t1/bnd_{speed,flow}_cv.txt and the row cache."""
import sys, copy, time, pickle, numpy as np, lightgbm as lgb, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, r"C:/Users/user/tfb/src")
from cachelib import load
from common import PANELS
from t1_features3 import grid_features3, rows3, base_of3
from t1_bnd import released, selector, build, PARAMS

OUT = "C:/Users/user/tfb/work/t1/"
MONTHS = ["2030-06", "2030-07", "2030-08", "2030-09", "2030-10", "2030-11", "2030-12", "2031-01"]


def month_slice(P, mon):
    idx = np.flatnonzero([s.startswith(mon) for s in P.times]); a, b = idx[0], idx[-1] + 1
    Q = copy.copy(P)
    for k in ("times", "V", "Q", "O", "P", "E", "regime", "target", "blank", "R", "RE"):
        setattr(Q, k, getattr(P, k)[a:b])
    Q.T = b - a; Q.day = np.array([t[:10] for t in Q.times[::288]])
    return Q


rows = {"tr": [], "va": []}; names = None; t0 = time.time()
for p in PANELS:
    P0 = load(p, "train")
    for mon in MONTHS:
        Pm = month_slice(P0, mon)
        F, _, _, _ = grid_features3(Pm); bv, bq = base_of3(F)
        V, Q = released(Pm, Pm.V, Pm.Q)
        sel, cc, ff = selector(Pm, V)
        m = Pm.target & Pm.E & np.isfinite(Pm.V) & np.isfinite(Pm.Q) & sel
        r, c = np.nonzero(m)
        X, names = build(Pm, F, V, Q, r, c, cc, ff, rows3)
        yv = Pm.V[r, c] / Pm.vf[c] - bv[r, c]; yq = Pm.Q[r, c] / Pm.lanes[c] - bq[r, c]
        ok = np.isfinite(yv) & np.isfinite(yq)
        rows["va" if mon == "2031-01" else "tr"].append((X[ok], yv[ok].astype(np.float32), yq[ok].astype(np.float32)))
        del F
    print(p, f"{time.time()-t0:.0f}s rows tr {sum(len(a[0]) for a in rows['tr'])} va {sum(len(a[0]) for a in rows['va'])}", flush=True)
    del P0; load.cache_clear()
pickle.dump(dict(rows=rows, names=names), open(OUT + "bnd_rows.pkl", "wb"), protocol=4)
Xtr = np.vstack([a[0] for a in rows["tr"]]); Xva = np.vstack([a[0] for a in rows["va"]])
best = {}
for ch, yi in (("speed", 1), ("flow", 2)):
    ytr = np.concatenate([a[yi] for a in rows["tr"]]); yva = np.concatenate([a[yi] for a in rows["va"]])
    ds = lgb.Dataset(Xtr, ytr, feature_name=names); dv = lgb.Dataset(Xva, yva, reference=ds)
    m = lgb.train(PARAMS, ds, 4000, valid_sets=[dv], callbacks=[lgb.early_stopping(100, verbose=False)])
    best[ch] = m.best_iteration; m.save_model(OUT + f"bnd_{ch}_cv.txt")
    print(ch, "best_iter", m.best_iteration, "valid rmse", m.best_score["valid_0"]["l2"] ** 0.5, f"{time.time()-t0:.0f}s", flush=True)
# refit on Jun-Jan with the found rounds (+10%)
Xall = np.vstack([Xtr, Xva])
for ch, yi in (("speed", 1), ("flow", 2)):
    y = np.concatenate([np.concatenate([a[yi] for a in rows["tr"]]), np.concatenate([a[yi] for a in rows["va"]])])
    m = lgb.train(PARAMS, lgb.Dataset(Xall, y, feature_name=names), int(best[ch] * 1.1) + 10)
    m.save_model(OUT + f"bnd_{ch}_final.txt"); print(ch, "refit done", f"{time.time()-t0:.0f}s", flush=True)
pickle.dump(best, open(OUT + "bnd_best.pkl", "wb"))
