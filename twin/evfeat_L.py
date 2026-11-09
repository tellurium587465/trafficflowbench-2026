"""Matched-base onset check in the law-calibrated twin (LightGBM proxy of the NN, evfeat_exp.py features M1 = dynamics +
site memory): train on world B alone vs B + L0..11 (calibrated variety), score L12..15 (held-out redraws of the same
base). The LB said adding randomized worlds hurts onset (about -0.017 IoU); the old twin said +0.07..+0.11. Prior =
site frequency of world B (the training configuration).
usage: python evfeat_L.py"""
import sys, numpy as np, pandas as pd, lightgbm as lgb
sys.argv = ["x"]
src = open(r"C:/Users/user/tfb/twin/evfeat_exp.py", encoding="utf-8").read().split('if __name__')[0]
src = src.replace('P = load(p, "twinA6"); truth, obs_q, _ = world_layers(P)', 'P = load(p, "twinB"); truth, obs_q, _ = world_layers(P)')
exec(src)
blocks = {}
for p in T2_PANELS:
    for w in ["B"] + [f"L{k}" for k in range(16)]:
        b = build_world(p, w)
        if b is not None:
            blocks.setdefault(w, []).append(b)
print("built", {k: sum(len(b[3]) for b in v) for k, v in blocks.items() if k in ("B", "L0", "L15")}, flush=True)
def XY(ws):
    bl = [b for w in ws for b in blocks.get(w, [])]
    return pd.DataFrame(np.vstack([b[0] for b in bl]), columns=COLS), np.concatenate([b[1] for b in bl]), bl
prm = dict(objective="binary", learning_rate=0.05, num_leaves=63, min_data_in_leaf=200, feature_fraction=0.8, bagging_fraction=0.8,
           bagging_freq=1, num_threads=16, verbose=-1, seed=0)
cols = SETS["M1"]; models = {}
for nm, ws in (("B", ["B"]), ("B+L", ["B"] + [f"L{k}" for k in range(12)])):
    X, Y, _ = XY(ws); models[nm] = lgb.train(prm, lgb.Dataset(X[cols], Y.astype(int)), 400)
rows = []
for w in [f"L{k}" for k in range(12, 16)]:
    for b in blocks.get(w, []):
        X = pd.DataFrame(b[0], columns=COLS); P = {k: m.predict(X[cols], num_threads=16) for k, m in models.items()}
        st = 0
        for L, dd in zip(b[3], b[2]["days"]):
            y = b[1][st:st + L]
            if y.any():
                r = dict(cl=b[2]["panel"] + w, day=dd)
                for k, pp in P.items():
                    s = bestset(pp[st:st + L]); r[k] = (s & y).sum() / (s | y).sum()
                rows.append(r)
            st += L
R = pd.DataFrame(rows); d = (R["B+L"] - R["B"]).to_numpy(); cl = R.cl.to_numpy(); u = np.unique(cl); idx = {x: np.flatnonzero(cl == x) for x in u}
rng = np.random.default_rng(0); se = np.std([d[np.concatenate([idx[x] for x in rng.choice(u, len(u))])].mean() for _ in range(400)])
print(f"held-out L worlds onsets {len(R)}: B-trained {R.B.mean():.4f}, B+L-trained {R['B+L'].mean():.4f}, diff {d.mean():+.4f} (SE {se:.4f}); days 1-5 {d[R.day.to_numpy() <= 5].mean():+.4f}")
