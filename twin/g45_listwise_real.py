"""G45: does the listwise gain of G44 (twin) hold on REAL data? sim2real.py blocks: real train months 0-5 (fit) and real
months 6-8 (test), morning first-queue onset events, features = dynamics (+ site prior). Pointwise binary LightGBM vs
LambdaRank (group = window, probabilities K * softmax(score / tau), tau from the training months only), same features.
Note: the blocks were built before the 10/13 month-left-out prior fix, so the real training rows' prior contains their
own month (affects both arms alike); the test months' prior comes from months 0-5 (out of sample).
usage: python g45_listwise_real.py"""
import os, sys, glob, pickle, numpy as np, pandas as pd, lightgbm as lgb
sys.path.insert(0, r"C:/Users/user/tfb/twin")
from adjuster import bestset
NT = int(os.environ.get("TFB_THREADS", "8"))
DYN = ["v1", "v6", "vmin", "q6", "qgrow", "vnb", "qdn"]; COLS = DYN + ["prior"]
B = {k: [] for k in ("real", "test")}
for f in glob.glob("E:/tfb_nn/sim2real/blocks_*.pkl"):
    d = pickle.load(open(f, "rb"))
    for k in B:
        B[k] += d[k]


def flat(bl):
    X, Y, G, P = [], [], [], []
    for (p, sp, Xs, Ys) in bl:
        for x, y in zip(Xs, Ys):
            if y.any():
                X.append(x); Y.append(y); G.append(len(y)); P.append(p)
    return X, Y, G, P


tr, te = flat(B["real"]), flat(B["test"])
print("train windows", len(tr[0]), "test windows", len(te[0]), flush=True)
Xtr = pd.DataFrame(np.vstack(tr[0]), columns=COLS); Ytr = np.concatenate(tr[1]).astype(int); K = float(np.mean([y.sum() for y in tr[1]]))
prm = dict(learning_rate=0.05, num_leaves=31, min_data_in_leaf=100, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
           num_threads=NT, verbose=-1, seed=0)
res = {}
for feats, tag in ((DYN, "M0"), (COLS, "M1")):
    pw = lgb.train(dict(prm, objective="binary"), lgb.Dataset(Xtr[feats], Ytr), 300)
    lr = lgb.train(dict(prm, objective="lambdarank", lambdarank_truncation_level=10), lgb.Dataset(Xtr[feats], Ytr, group=tr[2]), 300)
    def iou(s, y):
        u = (s | y).sum(); return 1.0 if u == 0 else (s & y).sum() / u
    def to_p(sc, tau):
        z = np.exp((sc - sc.max()) / tau); return np.clip(K * z / z.sum(), 0, 1)
    idx = np.random.default_rng(0).choice(len(tr[0]), min(400, len(tr[0])), replace=False)
    tau = max((0.5, 1.0, 2.0), key=lambda t: np.mean([iou(bestset(to_p(lr.predict(pd.DataFrame(tr[0][i], columns=COLS)[feats], num_threads=NT), t)), tr[1][i]) for i in idx]))
    for X, y, p in zip(te[0], te[1], te[3]):
        D = pd.DataFrame(X, columns=COLS)[feats]
        res.setdefault("panel", []).append(p) if tag == "M0" else None
        res.setdefault(f"{tag}_point", []).append(iou(bestset(pw.predict(D, num_threads=NT)), y))
        res.setdefault(f"{tag}_list", []).append(iou(bestset(to_p(lr.predict(D, num_threads=NT), tau)), y))
    print(tag, "tau", tau, flush=True)
R = pd.DataFrame(res); rng = np.random.default_rng(0); u = R.panel.unique(); idx = {x: np.flatnonzero(R.panel.to_numpy() == x) for x in u}
for tag in ("M0", "M1"):
    d = (R[f"{tag}_list"] - R[f"{tag}_point"]).to_numpy()
    se = np.std([d[np.concatenate([idx[x] for x in rng.choice(u, len(u))])].mean() for _ in range(1000)])
    print(f"{tag}: pointwise {R[f'{tag}_point'].mean():.4f}, listwise {R[f'{tag}_list'].mean():.4f}, diff {d.mean():+.4f} (panel-bootstrap SE {se:.4f})")
