"""G69 (10/22 night): the location-agnostic onset model, v2. Trained on ALL train-month onset windows with a T+30 queue
(3816, not only the 1494 selector-like ones), within-window percentile features only (no raw value -> no link
fingerprint; G68 "pr"). Two ways into the production onset blend (s46 spec) at T+30, then the evidence rule and set
(t2_m3v2_gate.score, k expectation), M3 dev onset windows (hold-out excluded):
  lin : p = (1 - a) p_NN + a p_phys
  qm  : rank transplant - the physics ranking gets the NN's own probability profile (the k-th ranked physics link gets
        the k-th largest NN probability of that window), p = (1 - a) p_NN + a p_qm. The two experts then agree on the set
        size and differ only in WHERE; hedging happens only where they disagree.
a in {0.2, 0.3, 0.5}; the choice cross-fitted on day parity; changed windows and rare / common truth split shown.
usage: python g69_phys_blend2.py"""
import sys, os, json, pickle
os.environ["TFB_T2CACHE"] = "2_noT_af"
sys.path.insert(0, r"C:/Users/user/tfb/src")
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


def feats(Xw):
    X = Xw[:, COLS].astype(float); P = np.full_like(X, 0.5)
    for c in range(X.shape[1]):
        v = X[:, c]; ok = np.isfinite(v)
        if ok.sum() >= 2 and np.nanstd(v) > 0:
            P[ok, c] = (rankdata(v[ok]) - 0.5) / ok.sum()
    return P


ON = [k for k in range(T.NWIN) if str(T.META["cond"][k]) == "queue_onset" and (T.LAB[k][0][5] & T.LAB[k][1][5]).any()]
Xs, ys = [], []
for k in ON:
    lab, el = T.LAB[k]; m = el[5]
    Xs.append(feats(np.asarray(XN[OFFN[k]:OFFN[k + 1]]))[m]); ys.append(lab[5][m].astype(int))
PRM = dict(objective="binary", learning_rate=0.05, num_leaves=31, min_data_in_leaf=50, feature_fraction=0.8,
           bagging_fraction=0.8, bagging_freq=1, lambda_l2=5.0, verbose=-1, num_threads=8, seed=7)
mdl = lgb.train(PRM, lgb.Dataset(np.vstack(Xs), np.concatenate(ys)), 400)
mdl.save_model(G.W + "g69_phys_onset.txt"); print(f"trained on {len(ON)} onset windows, {sum(len(y) for y in ys)} links", flush=True)

spec = json.load(open(r"C:/Users/user/tfb/src/t2_specs.json"))["s37"]["spec"]
parts = [(x.split(":")[0], float(x.split(":")[1]) if ":" in x else 1.0) for x in spec.split("+")]
PR = {t: pickle.load(open(G.W + f"probs_m3__{t}.pkl", "rb")) for t, _ in parts}
XM = np.load(f"{NN}/nn_test_m3/X.npy", mmap_mode="r"); OFFM = np.load(f"{NN}/nn_test_m3/off.npy")
POS = {str(w): i for i, w in enumerate(np.load(f"{NN}/nn_test_m3/window_id.npy", allow_pickle=True))}
WON = [w for w in G.WID if G.META[w]["cond"] == "queue_onset" and not m3v2.is_holdout(G.META[w]["day"])]
AS = (0.2, 0.3, 0.5); cols = {"nn": {}} | {f"{v}{a}": {} for v in ("lin", "qm") for a in AS}; PH = {}
for w in WON:
    m = G.META[w]; i = POS[w]; p = sum(wt * PR[t][w] for t, wt in parts) / sum(wt for _, wt in parts)
    ph = mdl.predict(feats(np.asarray(XM[OFFM[i]:OFFM[i + 1]]))); PH[w] = ph
    qm = np.empty_like(ph); qm[np.argsort(-ph)] = np.sort(p[5])[::-1]
    cols["nn"][w] = V.score(p, m)
    for a in AS:
        for v, src in (("lin", ph), ("qm", qm)):
            q = p.copy(); q[5] = (1 - a) * p[5] + a * src; cols[f"{v}{a}"][w] = V.score(q, m)
pickle.dump(PH, open(G.W + "probs_m3__g69phys_onset.pkl", "wb"))
D = V.frame(cols).dropna(subset=["nn"])
G64 = pd.read_csv(G.W + "g64_onset_rare_diag.csv").set_index("w"); D["rare"] = D.w.map(G64.rare_true)
D.to_csv(G.W + "g69_phys_blend2.csv", index=False)
for c in [c for c in cols if c != "nn"]:
    s = m3v2.summary(D, "nn", c, "private")
    print(m3v2.fmt(f"NN -> {c} (onset, dev)", s), f"| rare truth {(D[c] - D.nn)[D.rare == True].mean():+.4f}, common {(D[c] - D.nn)[D.rare == False].mean():+.4f}")
for v in ("lin", "qm"):
    cf = []
    for half in (0, 1):
        fit, ev = D[D.day % 2 == half], D[D.day % 2 != half]
        best = max(AS, key=lambda a: fit[f"{v}{a}"].mean()); cf.append((half, best, round(float(ev[f"{v}{best}"].mean() - ev["nn"].mean()), 4), len(ev)))
    print(f"{v}: cross-fit on day parity (fit half, chosen a, gain on the other half, n):", cf)
