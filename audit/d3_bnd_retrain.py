"""D3: the boundary specialist with a higher round cap (speed stopped at the 4000 cap in d1): lr 0.06, up to 12000
rounds, early stop on Jan; refit on Jun-Jan with best*1.1 -> work/t1/bnd2_{speed,flow}_final.txt."""
import sys, time, pickle, numpy as np, lightgbm as lgb
sys.path.insert(0, r"C:/Users/user/tfb/src")
from t1_bnd import PARAMS

OUT = "C:/Users/user/tfb/work/t1/"; t0 = time.time()
D = pickle.load(open(OUT + "bnd_rows.pkl", "rb")); rows, names = D["rows"], D["names"]
Xtr = np.vstack([a[0] for a in rows["tr"]]); Xva = np.vstack([a[0] for a in rows["va"]])
prm = dict(PARAMS, learning_rate=0.06, num_leaves=255, min_data_in_leaf=80); best = {}
for ch, yi in (("speed", 1), ("flow", 2)):
    ytr = np.concatenate([a[yi] for a in rows["tr"]]); yva = np.concatenate([a[yi] for a in rows["va"]])
    ds = lgb.Dataset(Xtr, ytr, feature_name=names); dv = lgb.Dataset(Xva, yva, reference=ds)
    m = lgb.train(prm, ds, 12000, valid_sets=[dv], callbacks=[lgb.early_stopping(150, verbose=False)])
    best[ch] = m.best_iteration
    print(ch, "best_iter", m.best_iteration, "valid rmse", round(m.best_score["valid_0"]["l2"] ** 0.5, 5), f"{time.time()-t0:.0f}s", flush=True)
Xall = np.vstack([Xtr, Xva]); del Xtr, Xva
for ch, yi in (("speed", 1), ("flow", 2)):
    y = np.concatenate([np.concatenate([a[yi] for a in rows["tr"]]), np.concatenate([a[yi] for a in rows["va"]])])
    m = lgb.train(prm, lgb.Dataset(Xall, y, feature_name=names), int(best[ch] * 1.1) + 10)
    m.save_model(OUT + f"bnd2_{ch}_final.txt"); print(ch, "refit done", f"{time.time()-t0:.0f}s", flush=True)
