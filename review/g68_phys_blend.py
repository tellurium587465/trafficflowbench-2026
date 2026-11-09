"""G68 (10/22 night): does the location-agnostic onset model of G67 help the production onset set on M3?
Binary LightGBM (per link: queued at T+30) on the train months' selector-like onset windows, features as G67: within-window
percentiles of the non-memory inputs, variant "pr+raw" (percentiles + raw values, as G67) and "pr" (percentiles only:
no raw value can carry a link fingerprint). Applied to the M3 onset windows; mixed into the production onset blend (s46
spec) at T+30: p = (1 - a) p_NN + a p_phys, then the production evidence rule and set (t2_m3v2_gate.score, k expectation).
Dev windows only (hold-out excluded); a in {0.1, 0.2, 0.3, 0.5}; the choice of a cross-fitted on day parity.
usage: python g68_phys_blend.py"""
import sys, os, json, pickle
os.environ["TFB_T2CACHE"] = "2_noT_af"
sys.path.insert(0, r"C:/Users/user/tfb/src"); sys.path.insert(0, r"C:/Users/user/tfb/review")
sys.argv = [sys.argv[0], "0"]
import numpy as np, pandas as pd, lightgbm as lgb
from scipy.stats import rankdata
from t2_eval_oof import T
from common import NN
import t2_m3_gate as G, t2_m3v2_gate as V, m3v2

XN = np.load(f"{NN}/nn_data_afdem/X.npy", mmap_mode="r"); OFFN = np.load(f"{NN}/nn_data_afdem/off.npy")
NAMES = [str(x) for x in np.load(f"{NN}/nn_data_afdem/names.npy", allow_pickle=True)]
DROP = {"prior_on", "prior_og", "pos", "vf", "capl", "lanes", "length", "w", "tod", "dow", "arr5", "cond"}
COLS = [i for i, n in enumerate(NAMES) if n not in DROP]
R66 = pd.read_csv("C:/Users/user/tfb/work/t2/g66_rare_context.csv")


def feats(Xw, raw):
    X = Xw[:, COLS].astype(float); P = np.full_like(X, 0.5)
    for c in range(X.shape[1]):
        v = X[:, c]; ok = np.isfinite(v)
        if ok.sum() >= 2 and np.nanstd(v) > 0:
            P[ok, c] = (rankdata(v[ok]) - 0.5) / ok.sum()
    return np.column_stack([P, X]) if raw else P


PRM = dict(objective="binary", learning_rate=0.05, num_leaves=31, min_data_in_leaf=50, feature_fraction=0.8,
           bagging_fraction=0.8, bagging_freq=1, lambda_l2=5.0, verbose=-1, num_threads=8, seed=7)
spec = json.load(open(r"C:/Users/user/tfb/src/t2_specs.json"))["s37"]["spec"]
parts = [(x.split(":")[0], float(x.split(":")[1]) if ":" in x else 1.0) for x in spec.split("+")]
PR = {t: pickle.load(open(G.W + f"probs_m3__{t}.pkl", "rb")) for t, _ in parts}
XM = np.load(f"{NN}/nn_test_m3/X.npy", mmap_mode="r"); OFFM = np.load(f"{NN}/nn_test_m3/off.npy")
POS = {str(w): i for i, w in enumerate(np.load(f"{NN}/nn_test_m3/window_id.npy", allow_pickle=True))}
WON = [w for w in G.WID if G.META[w]["cond"] == "queue_onset" and not m3v2.is_holdout(G.META[w]["day"])]
AS = (0.1, 0.2, 0.3, 0.5)
for raw in (True, False):
    Xs, ys = [], []
    for k in R66.k:
        lab, el = T.LAB[k]; m = el[5]
        Xs.append(feats(np.asarray(XN[OFFN[k]:OFFN[k + 1]]), raw)[m]); ys.append(lab[5][m].astype(int))
    mdl = lgb.train(PRM, lgb.Dataset(np.vstack(Xs), np.concatenate(ys)), 300)
    cols = {"nn": {}} | {f"a{a}": {} for a in AS} | {"phys": {}}
    for w in WON:
        m = G.META[w]; i = POS[w]; p = sum(wt * PR[t][w] for t, wt in parts) / sum(wt for _, wt in parts)
        ph = mdl.predict(feats(np.asarray(XM[OFFM[i]:OFFM[i + 1]]), raw))
        cols["nn"][w] = V.score(p, m)
        q = p.copy(); q[5] = ph; cols["phys"][w] = V.score(q, m)
        for a in AS:
            q = p.copy(); q[5] = (1 - a) * p[5] + a * ph; cols[f"a{a}"][w] = V.score(q, m)
    D = V.frame(cols).dropna(subset=["nn"]); tag = "pr+raw" if raw else "pr"
    D.to_csv(G.W + f"g68_phys_blend_{tag.replace('+', '_')}.csv", index=False)
    print(m3v2.fmt(f"[{tag}] physics alone vs NN (onset, dev)", m3v2.summary(D, "nn", "phys", "private")))
    for a in AS:
        print(m3v2.fmt(f"[{tag}] NN -> mix a={a} (onset, dev)", m3v2.summary(D, "nn", f"a{a}", "private")))
    cf = []
    for half in (0, 1):
        fit, ev = D[D.day % 2 == half], D[D.day % 2 != half]
        best = max(AS, key=lambda a: fit[f"a{a}"].mean()); cf.append((half, best, round(float(ev[f"a{best}"].mean() - ev["nn"].mean()), 4), len(ev)))
    print(f"[{tag}] cross-fit on day parity (fit half, chosen a, gain on the other half, n):", cf)
