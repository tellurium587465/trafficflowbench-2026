"""G44 (audit C, minimal re-test): is 'evidence does not help' / 'learned fusion loses to the rule' an artefact of the
pointwise log-loss + ratio-approximation decoder? Onset site choice is a within-window competition, so learn it listwise.
Data: evfeat_exp.py blocks (release-aligned blanks, one onset event per morning). Train = R worlds whose template month
is 2030-06..2030-11; test = U worlds (templates 2030-12..2031-02) -> no shared template days.
  a  M1 (dynamics + site prior) binary log-loss, x0.25 for links without evidence (the production rule)
  b  M3 (+ evidence features) binary log-loss
  c  M3 LambdaRank (group = window); probabilities = K * softmax(score / tau), K = mean positives per training window,
     tau chosen on the TRAINING worlds from {0.5, 1, 2} by IoU (no test information)
  d  M1 LambdaRank (separates the listwise effect from the evidence effect)
Decoder: best_set on the (T+30) probabilities. Score: IoU, days 1-5 and all days, cluster bootstrap (panel x world).
Threads: TFB_THREADS (default 8).
  a2 / b2  pointwise models with the same within-window mass normalisation as c/d (K * p / sum p): separates the
           ranking effect (needs retraining) from the normalisation effect (a post-hoc decoder change)
usage: python g44_listwise_evidence.py"""
import os, sys, glob, json, pickle, numpy as np, pandas as pd, lightgbm as lgb
sys.path.insert(0, r"C:/Users/user/tfb/twin")
from adjuster import bestset
OUT = "E:/tfb_nn/evfeat/"; NT = int(os.environ.get("TFB_THREADS", "8"))
COLS = ["v1", "v6", "vmin", "q6", "qgrow", "vnb", "qdn", "prior", "e_any", "e_head", "e1", "nd"]
M1 = COLS[:8]; M3 = COLS


def blocks(pattern):
    return [b for f in sorted(glob.glob(OUT + pattern)) for b in pickle.load(open(f, "rb"))]


def template(p, w):
    return json.load(open(f"C:/Users/user/tfb/work/twin/{p}_twin{w}.json"))["template"]


def windows(bl):
    X, Y, G, meta = [], [], [], []
    for b in bl:
        st = 0
        for L, dd in zip(b[3], b[2]["days"]):
            y = b[1][st:st + L]
            if y.any():
                X.append(b[0][st:st + L]); Y.append(y); G.append(L); meta.append((b[2]["panel"], b[2]["world"], dd))
            st += L
    return X, Y, G, meta


tr = windows([b for b in blocks("R_*_*.pkl") if template(b[2]["panel"], b[2]["world"]) < "2030-12"])
te = windows(blocks("U.pkl"))
print("train windows", len(tr[0]), "test windows", len(te[0]), flush=True)
Xtr = pd.DataFrame(np.vstack(tr[0]), columns=COLS); Ytr = np.concatenate(tr[1]).astype(int)
prm = dict(learning_rate=0.05, num_leaves=63, min_data_in_leaf=200, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
           num_threads=NT, verbose=-1, seed=0)
mod = {"a": lgb.train(dict(prm, objective="binary"), lgb.Dataset(Xtr[M1], Ytr), 400),
       "b": lgb.train(dict(prm, objective="binary"), lgb.Dataset(Xtr[M3], Ytr), 400),
       "c": lgb.train(dict(prm, objective="lambdarank", lambdarank_truncation_level=10), lgb.Dataset(Xtr[M3], Ytr, group=tr[2]), 400),
       "d": lgb.train(dict(prm, objective="lambdarank", lambdarank_truncation_level=10), lgb.Dataset(Xtr[M1], Ytr, group=tr[2]), 400)}
K = float(np.mean([y.sum() for y in tr[1]]))


def probs(name, X):
    D = pd.DataFrame(X, columns=COLS); cols = M1 if name[0] in ("a", "d") else M3
    s = mod[name[0]].predict(D[cols], num_threads=NT)
    return s


def to_p(s, tau):
    z = np.exp((s - s.max()) / tau); return np.clip(K * z / z.sum(), 0, 1)


def iou(sel, y):
    u = (sel | y).sum(); return 1.0 if u == 0 else (sel & y).sum() / u


def score_set(name, X, y, tau=None):
    s = probs(name, X)
    if name == "a":
        D = pd.DataFrame(X, columns=COLS); ev = D.e_any.to_numpy(); nd = D.nd.to_numpy()[0]
        p = s * (np.where(ev > 0, 1.0, 0.25) if nd > 0 and np.isfinite(ev).all() else 1.0)
    elif name == "b":
        p = s
    elif name in ("a2", "b2"):                                    # pointwise model, mass normalised within the window
        p = np.clip(K * s / max(s.sum(), 1e-9), 0, 1)
    else:
        p = to_p(s, tau)
    return iou(bestset(p), y)


taus = {}
for name in ("c", "d"):                                          # tau from the training worlds only (a 500-window sample)
    idx = np.random.default_rng(0).choice(len(tr[0]), min(500, len(tr[0])), replace=False)
    taus[name] = max((0.5, 1.0, 2.0), key=lambda t: np.mean([score_set(name, tr[0][i], tr[1][i], t) for i in idx]))
print("tau chosen on train:", taus, "K", round(K, 2), flush=True)
rows = []
for X, y, (p, w, dd) in zip(te[0], te[1], te[3]):
    r = dict(cl=p + w, day=dd)
    for name in ("a", "b", "c", "d", "a2", "b2"):
        r[name] = score_set(name, X, y, taus.get(name))
    rows.append(r)
R = pd.DataFrame(rows); rng = np.random.default_rng(0)
for lab, m in (("days 1-5", R.day <= 5), ("all days", R.day >= 1)):
    S = R[m]; u = S.cl.unique(); idx = {x: np.flatnonzero(S.cl.to_numpy() == x) for x in u}
    msg = []
    for c in ("b", "c", "d", "a2", "b2"):
        d = (S[c] - S["a"]).to_numpy(); se = np.std([d[np.concatenate([idx[x] for x in rng.choice(u, len(u))])].mean() for _ in range(1000)])
        msg.append(f"{c}-a {d.mean():+.4f} (SE {se:.4f})")
    print(f"{lab:9s} n {len(S):4d}: a {S.a.mean():.4f} | " + " | ".join(msg))
