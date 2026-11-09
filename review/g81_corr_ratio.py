"""G81 (INSTRUMENT_V3_SPEC §2.1 (a), 10/23): history-coverage correction ratios on the train months, onset first.
Selector-like train windows (G80: nn_test_trsel_off = official-type history, nn_test_trsel_m3 = M3-type history with the
day's Task 1 mask on T-12..T-1). Out-of-month NN = the month-CV fold models cvf_attnglob_s{0,1}_fold<group> (each window
predicted by the fold that did not train on its month), seeds averaged. Per condition:
  NN alone (expected-IoU set, onset at T+30, ongoing all rows, eligible cells)
  + evidence rule (onset: links unseen in rows -6..+11 around the slot on the k = min(3, day - 1) earlier days of the same
    month, +-2 links, x 0.25; evidence from the masked layer, identical in both conditions = the evidence axis)
  + physics onset (G69 recipe, trained on the onset windows of the other month groups, a = 0.3)
Effect = method - NN; ratio r = effect(official-type) / effect(M3-type); by the day's mask regime (0/1/2) for M3-type.
SE: window bootstrap (2000). Output work/t2/g81_corr_ratio.csv.
usage: python g81_corr_ratio.py          (after the fold models exist)"""
import sys, os, pickle, subprocess
os.environ["TFB_T2CACHE"] = "2_noT_af"
sys.path.insert(0, r"C:/Users/user/tfb/src")
sys.argv = [sys.argv[0], "0"]
import numpy as np, pandas as pd, lightgbm as lgb
from scipy.stats import rankdata
from t2_eval_oof import T
from t2_model import best_set
from cachelib import load
from common import NN

W = "C:/Users/user/tfb/work/t2/"; SRC = "C:/Users/user/tfb/src"
META = pickle.load(open(W + "trsel_meta.pkl", "rb"))          # (k, panel, t, cond, month, regime)
GROUP = {**{m: "2030-06" for m in ("2030-06", "2030-07", "2030-08")}, **{m: "2030-09" for m in ("2030-09", "2030-10", "2030-11")},
         **{m: "2030-12" for m in ("2030-12", "2031-01", "2031-02")}}
TAGS = [f"cvf_attnglob_s{s}_fold{g}" for s in (0, 1) for g in ("2030-06", "2030-09", "2030-12")]
for c in ("off", "m3"):                                          # CPU inference (seconds per model)
    todo = [t for t in TAGS if not os.path.exists(W + f"probs_trsel{c}__{t}.pkl")]
    if todo:
        subprocess.run([sys.executable, "t2_nn_cpu_infer.py", ",".join(todo), f"nn_test_trsel_{c}", f"trsel{c}"], cwd=SRC, check=True)
PR = {c: {t: pickle.load(open(W + f"probs_trsel{c}__{t}.pkl", "rb")) for t in TAGS} for c in ("off", "m3")}

# evidence from the masked layer (as released in a test month), same-month earlier days
QM = {}
def qmask(p):
    if p not in QM:
        P = load(p, "train"); V = np.where(P.blank, np.nan, P.V); QM[p] = np.isfinite(V) & (V <= P.vcut[None, :]) & P.E
    return QM[p]
def dil(s, r):
    o = s.copy()
    for d in range(1, r + 1):
        o[d:] |= s[:-d]; o[:-d] |= s[d:]
    return o
def seen(p, t, dom):
    q = qmask(p); s = np.zeros(q.shape[1], bool); nd = 0
    for back in range(1, min(3, dom - 1) + 1):
        dd = t - back * 288
        if dd - 6 < 0:
            break
        s |= q[dd - 6:dd + 12].any(axis=0); nd += 1
    return s, nd

# physics onset model per month group (trained on the other groups' onset windows, G69 recipe)
NAMES = [str(x) for x in np.load(f"{NN}/nn_data_afdem/names.npy", allow_pickle=True)]
COLS = [i for i, n in enumerate(NAMES) if n not in {"prior_on", "prior_og", "pos", "vf", "capl", "lanes", "length", "w", "tod", "dow", "arr5", "cond"}]
def feats(Xw):
    X = Xw[:, COLS].astype(float); P_ = np.full_like(X, 0.5)
    for c in range(X.shape[1]):
        v = X[:, c]; ok = np.isfinite(v)
        if ok.sum() >= 2 and np.nanstd(v) > 0:
            P_[ok, c] = (rankdata(v[ok]) - 0.5) / ok.sum()
    return P_
XN = np.load(f"{NN}/nn_data_afdem/X.npy", mmap_mode="r"); OFFN = np.load(f"{NN}/nn_data_afdem/off.npy")
mon = np.array([str(d)[:7] for d in T.META["date"]])
ON = [k for k in range(T.NWIN) if str(T.META["cond"][k]) == "queue_onset" and (T.LAB[k][0][5] & T.LAB[k][1][5]).any()]
PRM = dict(objective="binary", learning_rate=0.05, num_leaves=31, min_data_in_leaf=50, feature_fraction=0.8,
           bagging_fraction=0.8, bagging_freq=1, lambda_l2=5.0, verbose=-1, num_threads=8, seed=7)
PHY = {}
for g in ("2030-06", "2030-09", "2030-12"):
    tr = [k for k in ON if GROUP[mon[k]] != g]
    X = np.vstack([feats(np.asarray(XN[OFFN[k]:OFFN[k + 1]]))[T.LAB[k][1][5]] for k in tr]); y = np.concatenate([T.LAB[k][0][5][T.LAB[k][1][5]] for k in tr]).astype(int)
    PHY[g] = lgb.train(PRM, lgb.Dataset(X, y), 400); print("physics model for group", g, "trained", flush=True)
DX = {c: (np.load(f"{NN}/nn_test_trsel_{c}/X.npy", mmap_mode="r"), np.load(f"{NN}/nn_test_trsel_{c}/off.npy")) for c in ("off", "m3")}

rows = []
for i, (k, p, t, cond, month, regime) in enumerate(META):
    lab, el = T.LAB[k]; g = GROUP[month]; dom = int(str(T.META["date"][k])[8:10])
    r = dict(k=k, cond=cond, month=month, regime=regime)
    for c in ("off", "m3"):
        pr = sum(PR[c][f"cvf_attnglob_s{s}_fold{g}"][f"TR_{k}"] for s in (0, 1)) / 2
        if cond == "queue_onset":
            y, e = lab[5].astype(bool), el[5].astype(bool)
            iou = lambda p5: (lambda sel: (lambda u: 1.0 if u == 0 else float((sel & y & e).sum() / u))(((sel | y) & e).sum()))(best_set(p5, e))
            r[f"nn_{c}"] = iou(pr[5])
            s_, nd = seen(p, t, dom); pe = pr[5] * np.where(dil(s_, 2), 1.0, 0.25) if nd else pr[5]
            r[f"ev_{c}"] = iou(pe)
            a_, b_ = DX[c][1][i], DX[c][1][i + 1]; ph = PHY[g].predict(feats(np.asarray(DX[c][0][a_:b_])))
            r[f"phys_{c}"] = iou(0.7 * pr[5] + 0.3 * ph); r[f"evphys_{c}"] = iou(0.7 * pe + 0.3 * ph)
        else:
            y, e = lab.astype(bool), el.astype(bool); sel = best_set(pr.ravel(), e.ravel()).reshape(pr.shape)
            u = ((sel | y) & e).sum(); r[f"nn_{c}"] = 1.0 if u == 0 else float((sel & y & e).sum() / u)
    rows.append(r)
R = pd.DataFrame(rows); R.to_csv(W + "g81_corr_ratio.csv", index=False)
rng = np.random.default_rng(0)
def boot(x):
    x = np.asarray(x, float); return float(np.std([x[rng.integers(0, len(x), len(x))].mean() for _ in range(2000)]))
for cond in ("queue_onset", "queue_ongoing"):
    g = R[R.cond == cond]
    print(f"== {cond} (n {len(g)}): NN official-type {g.nn_off.mean():.4f} / M3-type {g.nn_m3.mean():.4f} (diff {(g.nn_m3 - g.nn_off).mean():+.4f} +- {boot(g.nn_m3 - g.nn_off):.4f})")
    if cond == "queue_onset":
        for mth in ("ev", "phys", "evphys"):
            d_off = g[f"{mth}_off"] - g.nn_off; d_m3 = g[f"{mth}_m3"] - g.nn_m3
            print(f"  {mth}: effect official-type {d_off.mean():+.4f} +- {boot(d_off):.4f} | M3-type {d_m3.mean():+.4f} +- {boot(d_m3):.4f} | r = {d_off.mean() / d_m3.mean() if d_m3.mean() != 0 else float('nan'):.2f}"
                  + " | M3-type by regime " + ", ".join(f"{rg}: {(g[g.regime == rg][f'{mth}_m3'] - g[g.regime == rg].nn_m3).mean():+.4f}" for rg in (0, 1, 2)))
