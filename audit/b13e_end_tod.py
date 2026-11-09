"""B13e (10/29, lane 11): how much does the time of day add to 'the jam ends within 30 min' beyond its age?
Training-month events at the known sites (as B13d). Rows = every row of each jam's life. LightGBM, leave-one-month-out
(9 folds), AUC and log loss of: age | age + site | age + site + tod | age + site + tod + start tod."""
import sys, numpy as np, pandas as pd, lightgbm as lgb
sys.path.insert(0, r"C:/Users/user/tfb/src")
from common import WORK
from sklearn.metrics import roc_auc_score, log_loss
D = pd.read_parquet(WORK + "/b13_events.parquet"); S = pd.read_parquet(WORK + "/b13_sites.parquet")
SID = {k: i for i, k in enumerate(zip(S.panel, S["head"]))}
D = D[D.raw0 & (D.dur >= 12)]; D = D[np.array([k in SID for k in zip(D.panel, D["head"])])]
rows = []
for e in D.itertuples():
    sid = SID[(e.panel, e.head)]
    for a in range(1, e.dur):
        rows.append((sid, a, (e.tod + a) % 288, e.tod, e.dur - a <= 6, e.day[:7]))
R = pd.DataFrame(rows, columns=["site", "age", "tod", "tod0", "y", "ym"]); print("rows", len(R), "positive", round(R.y.mean(), 3))
SETS = {"age": ["age"], "age+site": ["age", "site"], "age+site+tod": ["age", "site", "tod"], "age+site+tod+tod0": ["age", "site", "tod", "tod0"]}
prm = dict(objective="binary", learning_rate=0.05, num_leaves=31, min_data_in_leaf=200, verbose=-1, num_threads=8)
for nm, f in SETS.items():
    pr = np.zeros(len(R))
    for ym in R.ym.unique():
        tr, te = R.ym != ym, R.ym == ym
        m = lgb.train(prm, lgb.Dataset(R.loc[tr, f], R.y[tr], categorical_feature=["site"] if "site" in f else []), 300)
        pr[te.values] = m.predict(R.loc[te, f])
    print(f"{nm:20s} AUC {roc_auc_score(R.y, pr):.3f} | log loss {log_loss(R.y, pr):.4f}")
