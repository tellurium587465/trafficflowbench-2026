"""Information audit G10 (user request: is information left out?): the released ramp observations are not T2 inputs.
Value test by k-fold stacking on the onset windows (selector-like train set): label = link queued at T+30; base =
no-memory NN OOF probability; + ramp features at T (rows <= T-1, valid cells only): on/off ramp flow (last 15 min mean,
change vs the 45 min before), off/on ramp flow over mainline flow. No static attachment flags (link fingerprint).
LightGBM stacker trained on two month folds, scored on the third; per-window IoU (expected-IoU set at T+30) and AUC,
with and without the ramp features."""
import sys, numpy as np, pandas as pd, lightgbm as lgb
sys.path.insert(0, r"C:/Users/user/tfb/src")
import t2_eval_surprise as S
from t2_eval_oof import T, load_oof
from cachelib import load

oof = load_oof("q25_nomem_s3+q25_nomem_s4"); PC = {}
FOLD = {**{m: 0 for m in ("2030-06", "2030-07", "2030-08")}, **{m: 1 for m in ("2030-09", "2030-10", "2030-11")},
        **{m: 2 for m in ("2030-12", "2031-01", "2031-02")}}
rows, meta = [], []
for k in S.SEL:
    if k not in oof:
        continue
    p = str(T.META["panel"][k]); t = int(T.META["t"][k])
    if p not in PC:
        P = load(p, "train"); onv, offv, onf, offf = P.ramp_validity()
        PC[p] = (np.where(onv, onf, np.nan), np.where(offv, offf, np.nan), np.where(P.E, P.Q, np.nan))
    on, off, Q = PC[p]
    def m(a, lo, hi):
        with np.errstate(all="ignore"):
            return np.nanmean(a[max(0, t + lo):t + hi], axis=0)
    on3, on9, off3, off9, q3 = m(on, -3, 0), m(on, -12, -3), m(off, -3, 0), m(off, -12, -3), m(Q, -3, 0)
    lab, el = T.LAB[k]
    X = np.column_stack([oof[k][-1], on3, on3 - on9, off3, off3 - off9, off3 / np.maximum(q3, 1), on3 / np.maximum(q3, 1)])
    for i in range(X.shape[0]):
        rows.append(X[i]); meta.append((k, i, int(lab[-1][i]), FOLD[str(T.META["date"][k])[:7]]))
X = np.array(rows, np.float32); M = pd.DataFrame(meta, columns=["k", "i", "y", "fold"])
prm = dict(objective="binary", learning_rate=0.05, num_leaves=31, min_data_in_leaf=200, feature_fraction=0.9, verbose=-1, num_threads=8)
pred = {"base": np.zeros(len(M)), "stack_noramp": np.zeros(len(M)), "stack_ramp": np.zeros(len(M))}
pred["base"] = X[:, 0]
for f in range(3):
    tr, te = (M.fold != f).to_numpy(), (M.fold == f).to_numpy()
    for name, cols in (("stack_noramp", [0]), ("stack_ramp", list(range(X.shape[1])))):
        b = lgb.train(prm, lgb.Dataset(X[tr][:, cols], M.y[tr]), 200)
        pred[name][te] = b.predict(X[te][:, cols])


def bestset_last(p, n):
    o = np.argsort(-p); cs = np.cumsum(p[o]); kk = np.arange(1, n + 1); kb = int(np.argmax(cs / (kk + p.sum() - cs))) + 1
    s = np.zeros(n, bool); s[o[:kb]] = True; return s


from sklearn.metrics import roc_auc_score
for name, pr in pred.items():
    ious = []
    for k, g in M.groupby("k"):
        lab, el = T.LAB[k]; lab = lab.astype(bool); el = el.astype(bool)
        s = np.zeros_like(lab); s[-1] = bestset_last(pr[g.index.to_numpy()], len(g))
        u = ((s | lab) & el).sum(); ious.append(1.0 if u == 0 else (s & lab & el).sum() / u)
    print(f"{name:13s} held-out onset IoU {np.mean(ious):.4f}  AUC {roc_auc_score(M.y, pr):.4f}")
