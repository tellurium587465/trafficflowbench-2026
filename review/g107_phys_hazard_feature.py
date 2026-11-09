"""G107 (10/29, strategy lane 4, X): per-link stochastic capacity (Brilon-type Weibull, G103: fitted on the training months only,
site-specific lambda where a site broke down >= 50 times, shrunk to its geometry class otherwise) as features of the physics
onset ranker. The generator keeps capacity fixed across months, so "this site breaks at this flow" transfers to a new month.
Features per link of a window: hazard at the flow at T and at the window's maximum flow, h = 1 - exp(-(q_lane / lambda)^gamma),
raw and as within-window percentiles, appended to the ranker's percentile features (t2_phys_onset.feats).
Identity: the base ranker retrained here reproduces physon_base.txt on the M3 windows (checked below).
Judged on M3 v3 dev onset windows in the final A's onset flow (q24f 8 seeds CF5 + ranker mixed at T+30 with the month-adaptive
a + evidence rule + decoder), as G93's "phys" column. Pre-declared (ledger): k = 1, house rule (point > 0 and no harm <= -2 SE).
Writes work/t2/g107_phys_hazard.csv for judge.py --cond queue_onset.
usage: TFB_M3VER=v3 python g107_phys_hazard_feature.py"""
import os, sys, pickle
os.environ.setdefault("TFB_M3VER", "v3")
sys.path.insert(0, r"C:/Users/user/tfb/src"); argv0 = sys.argv[0]
import numpy as np, lightgbm as lgb
from scipy.stats import rankdata
import t2_phys_onset as PO
from cachelib import load
sys.argv = [argv0, "0,1,2,3"]
import t2_m3_gate as G, t2_m3v2_gate as V, m3v2
from common import NN

W = G.W; VER = G.VER; PRE = G.PRE
HZ = pickle.load(open(W + "g103_hazard.pkl", "rb")); LAM, GAM = HZ["LAMJ"], HZ["GAMJ"]
NAMES = PO.NAMES; IQ, IQM = NAMES.index("last_q"), NAMES.index("q_max")
STAT = {}


def static(p):
    if p not in STAT:
        P = load(p, "validation"); STAT[p] = (P.cap / P.lanes); load.cache_clear()
    return STAT[p]


def pct(v):
    out = np.full_like(v, 0.5); ok = np.isfinite(v)
    if ok.sum() >= 2 and np.nanstd(v) > 0:
        out[ok] = (rankdata(v[ok]) - 0.5) / ok.sum()
    return out


def hazard_feats(Xw, p):
    capl = static(p); lam, gam = LAM[p], GAM[p]
    h = []
    for i in (IQ, IQM):
        q = np.nan_to_num(np.asarray(Xw[:, i], float), nan=0.0).clip(0) * capl
        h.append(1 - np.exp(-(q / lam) ** gam))
    hT, hM = h
    return np.column_stack([hT, hM, pct(hT), pct(hM)])


def rows(d, pan):
    X = np.load(f"{NN}/{d}/X.npy", mmap_mode="r"); Y = np.load(f"{NN}/{d}/Y.npy", mmap_mode="r"); E = np.load(f"{NN}/{d}/EL.npy", mmap_mode="r")
    off = np.load(f"{NN}/{d}/off.npy"); cond = np.load(f"{NN}/{d}/cond.npy", allow_pickle=True); A, B, ys = [], [], []
    for k in np.flatnonzero(cond == "queue_onset"):
        a, b = off[k], off[k + 1]; y = np.asarray(Y[a:b, 5]).astype(bool); e = np.asarray(E[a:b, 5]).astype(bool)
        if (y & e).any():
            Xw = np.asarray(X[a:b]); A.append(PO.feats(Xw)[e]); B.append(hazard_feats(Xw, str(pan[k]))[e]); ys.append(y[e].astype(int))
    return np.vstack(A), np.vstack(B), np.concatenate(ys)


pan = np.load(f"{NN}/nn_data_afdem/panel.npy", allow_pickle=True)
Fa, Fh, y = rows("nn_data_afdem", pan); print("training rows", len(y), "positives", int(y.sum()), flush=True)
base = lgb.train(PO.PRM, lgb.Dataset(Fa, y), 400)
haz = lgb.train(PO.PRM, lgb.Dataset(np.column_stack([Fa, Fh]), y), 400)
saved = lgb.Booster(model_file=W + "physon_base.txt")
CF5 = (("attnglob", 2.0), ("attnwp05", 2.0), ("attn1", 2.0), ("drop1", 1.0), ("r13", 1.0))
ON = [(f"q24f_{c}_s{s}", w) for s in range(8) for c, w in CF5]; PRO = {t: pickle.load(open(W + f"probs_{PRE}__{t}.pkl", "rb")) for t, _ in ON}
X = np.load(f"{NN}/nn_test_m3{VER}/X.npy", mmap_mode="r"); OFF = np.load(f"{NN}/nn_test_m3{VER}/off.npy")
ev = PO.month_events("validation", PO.rarity_table())
cols = {"base": {}, "haz": {}}; dmax = 0.0
for i, w in enumerate(G.WID):
    m = G.META[w]
    if m["cond"] != "queue_onset" or m3v2.is_holdout(np.array([m["day"]]))[0]:
        continue
    Xw = np.asarray(X[OFF[i]:OFF[i + 1]]); F0 = PO.feats(Xw); Fz = hazard_feats(Xw, m["panel"])
    pb, ph = base.predict(F0), haz.predict(np.column_stack([F0, Fz])); dmax = max(dmax, float(np.abs(pb - saved.predict(F0)).max()))
    pn = sum(wt * PRO[t][w] for t, wt in ON) / sum(wt for _, wt in ON); a = PO.adaptive_a(ev, m["day"])
    for nm, pr_ in (("base", pb), ("haz", ph)):
        p = pn.copy(); p[5] = (1 - a) * p[5] + a * pr_
        cols[nm][w] = sum(pr * G.iou(V.onset_pred(p, m, "cur", k), m) for k, pr in m3v2.kdist(m["day"]).items())
print(f"identity: retrained base vs physon_base.txt on the M3 onset windows: max |dp| {dmax:.2e}")
D = V.frame(cols); D.to_csv(W + "g107_phys_hazard.csv", index=False)
dev = ~m3v2.is_holdout(D.day)
print(f"M3 {VER} dev onset windows {int(dev.sum())}: base {D.base[dev].mean():.4f} | + hazard features {D.haz[dev].mean():.4f}")
imp = haz.feature_importance("gain"); nb = Fa.shape[1]
print("gain share of the 4 hazard features: %.1f%%" % (100 * imp[nb:].sum() / imp.sum()))
