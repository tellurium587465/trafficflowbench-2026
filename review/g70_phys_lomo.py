"""G70 (10/22, strategy §16 (a)): does the location-agnostic onset model hurt the USUAL sites? Train months, large sample.
Physics model as G69 (within-window percentiles of the non-memory inputs, binary LightGBM, 400 rounds) trained
leave-one-month-out on all onset windows of the other 8 months; NN = month-CV OOF of the no-memory NN (q25_nomem seeds 3,4,
the s46-era baseline). On the selector-like onset windows with a T+30 queue: T+30 set by the expected-IoU decoder
(t2_model.best_set, eligible cells, no evidence rule: neutral on the train months, G4), IoU at T+30, for
p = (1 - a) p_NN + a p_phys, a in {0, 0.1, 0.2, 0.3, 0.5}. Paired differences vs a = 0, SE clustered by month x panel,
split by new-site onsets (leave-month-out rarity <= 0.1, G66) and usual sites.
usage: python g70_phys_lomo.py"""
import sys, os, pickle
os.environ["TFB_T2CACHE"] = "2_noT_af"
sys.path.insert(0, r"C:/Users/user/tfb/src")
sys.argv = [sys.argv[0], "0"]
import numpy as np, pandas as pd, lightgbm as lgb
from scipy.stats import rankdata
from t2_eval_oof import T
from t2_model import best_set
from common import NN

W = "C:/Users/user/tfb/work/t2/"
XN = np.load(f"{NN}/nn_data_afdem/X.npy", mmap_mode="r"); OFFN = np.load(f"{NN}/nn_data_afdem/off.npy")
NAMES = [str(x) for x in np.load(f"{NN}/nn_data_afdem/names.npy", allow_pickle=True)]
DROP = {"prior_on", "prior_og", "pos", "vf", "capl", "lanes", "length", "w", "tod", "dow", "arr5", "cond"}
COLS = [i for i, n in enumerate(NAMES) if n not in DROP]


def feats(Xw):
    X = Xw[:, COLS].astype(float); P = np.full_like(X, 0.5)
    for c in range(X.shape[1]):
        v = X[:, c]; ok = np.isfinite(v)
        if ok.sum() >= 2 and np.nanstd(v) > 0:
            P[ok, c] = (rankdata(v[ok]) - 0.5) / ok.sum()
    return P


mon = lambda k: str(T.META["date"][k])[:7]
ON = [k for k in range(T.NWIN) if str(T.META["cond"][k]) == "queue_onset" and (T.LAB[k][0][5] & T.LAB[k][1][5]).any()]
F = {k: feats(np.asarray(XN[OFFN[k]:OFFN[k + 1]])) for k in ON}
OOF = [pickle.load(open(W + f"oof_q25_nomem_s{s}.pkl", "rb")) for s in (3, 4)]
R66 = pd.read_csv(W + "g66_rare_context.csv"); NEW = dict(zip(R66.k, R66.new))
EV = [k for k in ON if T.META["is_sel"][k] and k in OOF[0] and k in NEW]
PRM = dict(objective="binary", learning_rate=0.05, num_leaves=31, min_data_in_leaf=50, feature_fraction=0.8,
           bagging_fraction=0.8, bagging_freq=1, lambda_l2=5.0, verbose=-1, num_threads=8, seed=7)
AS = (0.0, 0.1, 0.2, 0.3, 0.5); rows = []
for mo in sorted({mon(k) for k in EV}):
    tr = [k for k in ON if mon(k) != mo]
    X = np.vstack([F[k][T.LAB[k][1][5]] for k in tr]); y = np.concatenate([T.LAB[k][0][5][T.LAB[k][1][5]] for k in tr]).astype(int)
    mdl = lgb.train(PRM, lgb.Dataset(X, y), 400)
    for k in [k for k in EV if mon(k) == mo]:
        lab, el = T.LAB[k]; yk, ek = lab[5].astype(bool), el[5].astype(bool)
        pn = (OOF[0][k][5] + OOF[1][k][5]) / 2; ph = mdl.predict(F[k])
        r = dict(k=k, month=mo, panel=str(T.META["panel"][k]), new=bool(NEW[k]))
        for nm, sc in (("rk_nn", pn), ("rk_ph", ph)):                       # rank of the best true eligible link
            sc_ = np.where(ek, sc, -1.0); o = np.argsort(-sc_); rr = np.empty(len(o), int); rr[o] = np.arange(len(o))
            r[nm] = int(rr[yk & ek].min())
        for a in AS:
            sel = best_set((1 - a) * pn + a * ph, ek); u = ((sel | yk) & ek).sum()
            r[f"a{a}"] = 1.0 if u == 0 else float((sel & yk & ek).sum() / u)
        rows.append(r)
    print(mo, "done", flush=True)
D = pd.DataFrame(rows); D.to_csv(W + "g70_phys_lomo.csv", index=False)


def paired(g, c):
    d = g[c] - g["a0.0"]; cl = g.month + "|" + g.panel
    cm = d.groupby(cl).sum(); n = len(d); se = np.sqrt(len(cm) / max(len(cm) - 1, 1) * ((cm - cm.index.map(cl.value_counts()) * d.mean()) ** 2).sum()) / n
    return d.mean(), se


print(f"train-month selector-like onset windows {len(D)} (new-site {D.new.sum()}); NN alone IoU {D['a0.0'].mean():.4f}")
for a in AS[1:]:
    out = []
    for name, g in (("all", D), ("usual", D[~D.new]), ("new-site", D[D.new])):
        m_, s_ = paired(g, f"a{a}"); out.append(f"{name} {m_:+.4f} ({m_ / s_:+.1f} SE)" if s_ > 0 else f"{name} {m_:+.4f}")
    print(f"a={a}: " + " | ".join(out) + f" | windows changed {(D[f'a{a}'] != D['a0.0']).sum()}")
for name, g in (("usual", D[~D.new]), ("new-site", D[D.new])):
    print(f"{name}: top-1/3/5 NN {np.mean(g.rk_nn == 0):.2f}/{np.mean(g.rk_nn < 3):.2f}/{np.mean(g.rk_nn < 5):.2f} | "
          f"physics {np.mean(g.rk_ph == 0):.2f}/{np.mean(g.rk_ph < 3):.2f}/{np.mean(g.rk_ph < 5):.2f}")
