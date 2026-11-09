"""How much does ONE extra configuration help (the analogue of adding the validation month, a second real
configuration, to the training data of the private predictions)? Same features / models as evfeat_exp.py (M1 = dyn +
site memory). Train on A6 alone, A6 + one randomized world per panel (R0, R1 or R2 as the 'extra month', 3 draws),
A6 + 3 worlds, and all R; score T/U/S days 1-5 and all days (mean IoU at T+30).
usage: python evfeat_onemore.py"""
import sys, pickle, glob, numpy as np, pandas as pd, lightgbm as lgb
sys.argv = ["x"]
exec(open(r"C:/Users/user/tfb/twin/evfeat_exp.py", encoding="utf-8").read().split('if __name__')[0])
blocksR = [b for f in sorted(glob.glob(OUT + "R_*_*.pkl")) for b in pickle.load(open(f, "rb"))]
blocksA = pickle.load(open(OUT + "A6.pkl", "rb"))


def XY(blocks):
    return pd.DataFrame(np.vstack([b[0] for b in blocks]), columns=COLS), np.concatenate([b[1] for b in blocks])


prm = dict(objective="binary", learning_rate=0.05, num_leaves=63, min_data_in_leaf=200, feature_fraction=0.8,
           bagging_fraction=0.8, bagging_freq=1, num_threads=16, verbose=-1, seed=0)
cols = SETS["M1"]
train_sets = {"A6": blocksA}
for k in range(3):
    train_sets[f"A6+R{k}"] = blocksA + [b for b in blocksR if b[2]["world"] == f"R{k}"]
train_sets["A6+R0..2"] = blocksA + [b for b in blocksR if b[2]["world"] in ("R0", "R1", "R2")]
train_sets["A6+R0..11"] = blocksA + [b for b in blocksR if int(b[2]["world"][1:]) < 12]
train_sets["R all"] = blocksR
models = {}
for nm, bl in train_sets.items():
    X, Y = XY(bl); models[nm] = lgb.train(prm, lgb.Dataset(X[cols], Y.astype(int)), 400)
    print(nm, "windows", sum(len(b[3]) for b in bl), flush=True)
rows = []
for fam in ("T", "U", "S"):
    X, Y, win = stack(fam); P = {k: m.predict(X[cols], num_threads=16) for k, m in models.items()}
    for (st, L, p, w, dd) in win:
        y = Y[st:st + L]
        if not y.any():
            continue
        r = dict(fam=fam, day=dd, cl=p + w)
        for k, pp in P.items():
            s = bestset(pp[st:st + L]); r[k] = (s & y).sum() / (s | y).sum()
        rows.append(r)
R = pd.DataFrame(rows); ks = list(models)
print("all days:", R[ks].mean().round(4).to_dict())
print("days 1-5:", R[R.day <= 5][ks].mean().round(4).to_dict())
rng = np.random.default_rng(0); u = R.cl.unique(); idx = {x: np.flatnonzero(R.cl.to_numpy() == x) for x in u}
for k in ks[1:]:
    d = (R[k] - R["A6"]).to_numpy(); se = np.std([d[np.concatenate([idx[x] for x in rng.choice(u, len(u))])].mean() for _ in range(300)])
    print(f"{k:10s} - A6: {d.mean():+.4f} (SE {se:.4f})")
