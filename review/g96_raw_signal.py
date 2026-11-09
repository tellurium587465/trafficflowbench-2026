"""G96 (10/23, X3 need test before building the net; memory: measure the need first, minimal version): does the RAW
history (t2_raw_hist.py: 13 rows x v/vcut, q/cap, observed) carry ongoing signal beyond the engineered features?
LightGBM (binary, per link, target = queued at T+30 on eligible cells) on ongoing training windows, two feature sets with the
no-memory drops (prior / position / static / time-of-day / weekday out): E = the engineered columns, E+R = E + the 39 raw
values. Train = train months 2030-06..2030-12, test = 2031-01..02 (later months, a 25% window sample for speed); then the
same E / E+R models trained on all train months scored on the M3 v3 dev ongoing windows (observed eligible cells at T+30).
Paired: logloss and AUC differences with a window-cluster bootstrap SE. Decision (pre-declared, STATE 10/23 21:52): build
the X3 net only if E+R beats E on both by >= 2 SE in logloss; otherwise X3 stops here (the engineered summary already
carries the raw history's signal).
usage: python g96_raw_signal.py"""
import os, sys, pickle, numpy as np, lightgbm as lgb
os.environ.setdefault("TFB_M3VER", "v3")
sys.path.insert(0, r"C:/Users/user/tfb/src"); sys.argv = [sys.argv[0], "0"]
from sklearn.metrics import roc_auc_score
from common import NN
import t2_m3_gate as G, m3v2

names = [str(x) for x in np.load(f"{NN}/nn_data_afdem/names.npy", allow_pickle=True)]
DROP = {"prior_on", "prior_og", "pos", "vf", "capl", "lanes", "length", "w", "tod", "dow", "cond"}
EC = [i for i, n in enumerate(names) if n not in DROP]
PRM = dict(objective="binary", learning_rate=0.05, num_leaves=63, min_data_in_leaf=100, feature_fraction=0.8, bagging_fraction=0.8,
           bagging_freq=1, lambda_l2=5.0, verbose=-1, num_threads=8, seed=7)
rng = np.random.default_rng(0)


def rows(d, ks, step=5):
    X = np.load(f"{NN}/{d}/X.npy", mmap_mode="r"); R = np.load(f"{NN}/{d}/R.npy", mmap_mode="r")
    Y = np.load(f"{NN}/{d}/Y.npy", mmap_mode="r"); E = np.load(f"{NN}/{d}/EL.npy", mmap_mode="r"); off = np.load(f"{NN}/{d}/off.npy")
    Fe, Fr, y, g = [], [], [], []
    for k in ks:
        a, b = off[k], off[k + 1]; e = np.asarray(E[a:b, step]).astype(bool)
        if not e.any():
            continue
        Fe.append(np.asarray(X[a:b])[e][:, EC]); Fr.append(np.asarray(R[a:b], dtype=np.float32)[e].reshape(int(e.sum()), -1))
        y.append(np.asarray(Y[a:b, step])[e].astype(int)); g.append(np.full(int(e.sum()), k))
    return np.vstack(Fe), np.vstack(Fr), np.concatenate(y), np.concatenate(g)


def paired(y, pa, pb, g, nb=300):
    ll = lambda p: -(y * np.log(np.clip(p, 1e-6, 1)) + (1 - y) * np.log(np.clip(1 - p, 1e-6, 1)))
    d = ll(pa) - ll(pb)                                          # > 0: B better
    ug, inv = np.unique(g, return_inverse=True); s = np.bincount(inv, d); c = np.bincount(inv)
    bs = [s[i].sum() / c[i].sum() for i in (rng.integers(0, len(ug), len(ug)) for _ in range(nb))]
    return d.mean(), float(np.std(bs)), roc_auc_score(y, pb) - roc_auc_score(y, pa)


d_ = "nn_data_afdem"; cond = np.load(f"{NN}/{d_}/cond.npy", allow_pickle=True); date = np.load(f"{NN}/{d_}/date.npy", allow_pickle=True)
mon = np.array([str(x)[:7] for x in date]); og = np.flatnonzero(cond == "queue_ongoing")
tr = [k for k in og if mon[k] < "2031-01" and rng.random() < 0.25]; te = [k for k in og if mon[k] >= "2031-01" and rng.random() < 0.25]
Fe, Fr, y, _ = rows(d_, tr); Te, Tr, ty, tg = rows(d_, te)
print(f"train windows {len(tr)} cells {len(y)} | test windows {len(te)} cells {len(ty)} | positives {y.mean():.3f} / {ty.mean():.3f}", flush=True)
mE = lgb.train(PRM, lgb.Dataset(Fe, y), 400); mER = lgb.train(PRM, lgb.Dataset(np.hstack([Fe, Fr]), y), 400)
m_, s_, auc = paired(ty, mE.predict(Te), mER.predict(np.hstack([Te, Tr])), tg)
print(f"later train months: logloss gain of E+R over E {m_:+.5f} (SE {s_:.5f}, {m_ / s_:+.1f} SE), AUC {auc:+.4f}", flush=True)

# M3 v3 dev ongoing windows (labels on observed eligible cells at T+30, as the M3 gate scores)
allk = [k for k in og if rng.random() < 0.5]
Fe, Fr, y, _ = rows(d_, allk)
mE = lgb.train(PRM, lgb.Dataset(Fe, y), 400); mER = lgb.train(PRM, lgb.Dataset(np.hstack([Fe, Fr]), y), 400)
XM = np.load(f"{NN}/nn_test_m3{G.VER}/X.npy", mmap_mode="r"); RM = np.load(f"{NN}/nn_test_m3{G.VER}/R.npy", mmap_mode="r"); OFF = np.load(f"{NN}/nn_test_m3{G.VER}/off.npy")
Te, Tr, ty, tg = [], [], [], []
for i, w in enumerate(G.WID):
    m = G.META[w]
    if m["cond"] != "queue_ongoing" or m3v2.is_holdout(m["day"]):
        continue
    e = m["m"][:, 5].astype(bool) if m["m"].shape[0] != 6 else m["m"][5].astype(bool)
    yy = m["y"][:, 5] if m["y"].shape[0] != 6 else m["y"][5]
    a, b = OFF[i], OFF[i + 1]
    Te.append(np.asarray(XM[a:b])[e][:, EC]); Tr.append(np.asarray(RM[a:b], dtype=np.float32)[e].reshape(int(e.sum()), -1))
    ty.append(np.asarray(yy)[e].astype(int)); tg.append(np.full(int(e.sum()), i))
Te, Tr, ty, tg = np.vstack(Te), np.vstack(Tr), np.concatenate(ty), np.concatenate(tg)
m_, s_, auc = paired(ty, mE.predict(Te), mER.predict(np.hstack([Te, Tr])), tg)
print(f"M3 {G.VER} dev ongoing ({len(np.unique(tg))} windows, {len(ty)} cells): logloss gain of E+R over E {m_:+.5f} (SE {s_:.5f}, {m_ / s_:+.1f} SE), AUC {auc:+.4f}")
