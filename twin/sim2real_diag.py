"""Why does a twin-v4-trained onset model rank real onset links worse (within-window AUC 0.878 vs 0.971)? (10/14)
Per panel (no pooling across panels): train on the real months 0-5 and on 12 twin v4 worlds (months 0-5 templates),
test on the real months 6-8 of the same panel. Reports IoU / AUC per model, LightGBM gain importance, and the mean
feature values at positive and negative cells in real-train vs twin-train windows (what differs before an onset).
Reuses sim2real_v4.py's window / feature code (imported with its main loop skipped).
usage: python sim2real_diag.py [panels=D7_I10_E,D12_I5_N]"""
import sys, zlib, numpy as np, pandas as pd, lightgbm as lgb, warnings
warnings.simplefilter("ignore")
sys.path.insert(0, r"C:/Users/user/tfb/src"); sys.path.insert(0, r"C:/Users/user/tfb/twin")
src = open(r"C:/Users/user/tfb/twin/sim2real_v4.py", encoding="utf-8").read()
ns = {"__name__": "lib"}; exec(src.split("data = {\"REAL\"")[0], ns)                 # functions only
from cachelib import load
from onset_shape3_lib import raw_matrices
from ctm import junction_ratios
from t2_model import best_set
from scipy.stats import rankdata
import gen, gen_v4

NAMES = ["v1", "v3", "vmin", "q1", "q6", "qg", "v1_up1", "v1_dn1", "v1_up2", "v1_dn2", "dq_dn", "dq_up", "pos"]
PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=31, min_data_in_leaf=200, feature_fraction=0.9,
              bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, num_threads=8, verbose=-1, seed=0)
ns["COND"] = "onset"; ns["CAPW"][0] = 250


def stackd(L):
    X = np.vstack([x[e] for xs, ys, es, _ in L for x, e in zip(xs, es)]); y = np.concatenate([y[e] for xs, ys, es, _ in L for y, e in zip(ys, es)])
    return X, y.astype(np.float32)


for p in (sys.argv[1] if len(sys.argv) > 1 else "D7_I10_E,D12_I5_N").split(","):
    PT = load(p, "train"); D = PT.T // 288; PT.V, PT.Q, PT.E = raw_matrices(PT); ns["VCUT"] = PT.vcut
    months = sorted(set(str(x)[:7] for x in PT.day[:D])); keep = [d for d in range(D) if str(PT.day[d])[:7] in months[:6]]
    test = [d for d in range(D) if d not in set(keep)]; vf, cap = PT.vf.astype(float), PT.cap.astype(float)
    R = {}
    for tag, days in (("REAL", keep), ("TEST", test)):
        r = ns["month_rows"](days); R[tag] = [ns["collect"](PT.V[r], PT.Q[r], PT.E[r], len(days), vf, cap, tag, p)]
    S = ns["structure_k"](PT, keep); jr = np.isin(np.arange(PT.T) // 288, keep)
    jr_k = lambda P_, rows=None, _jr=jr: junction_ratios(P_, _jr); gen.junction_ratios = gen_v4.junction_ratios = jr_k
    base_lr = np.log(jr_k(PT)).mean(axis=(0, 1)); rng = np.random.default_rng([7, zlib.crc32(p.encode())]); R["V4"] = []
    for k in range(12):
        tm = months[k % 6]; days = [d for d in keep if str(PT.day[d]).startswith(tm)][:30]
        s_rat, s_up = ns["RATS"][rng.integers(3)], ns["UPS"][rng.integers(3)]
        cm = np.exp(gen.SIG_CAP * rng.standard_normal(PT.n)); rat = s_rat * np.abs(base_lr) * rng.standard_normal(PT.n - 1)
        upm = float(np.exp(s_up * rng.standard_normal())); r = ns["month_rows"](days); nanm = ~np.isfinite(PT.V[r])
        Vt, Qt, _ = gen_v4.simulate_v4(PT, days, rng, S, rat, upm, cm)
        Vt = Vt * (1 + gen.NOISE_V * rng.standard_normal(Vt.shape)); Qt = Qt * (1 + gen.NOISE_Q * rng.standard_normal(Qt.shape))
        Vt[nanm] = np.nan; Qt[nanm] = np.nan
        R["V4"].append(ns["collect"](Vt, Qt, PT.E[r], len(days), vf, cap, "V4", f"{p}V4{k}_"))
    print(f"{p}: windows real {len(R['REAL'][0][0])}, twin {sum(len(c[0]) for c in R['V4'])}, test {len(R['TEST'][0][0])}", flush=True)
    M = {}
    for name in ("REAL", "V4"):
        X, y = stackd(R[name]); M[name] = lgb.train(PARAMS, lgb.Dataset(X, y), 300)
        imp = M[name].feature_importance("gain"); top = np.argsort(-imp)[:6]
        print(f"  {name} model: positive share {y.mean():.4f}, top gain " + ", ".join(f"{NAMES[i]} {imp[i] / imp.sum():.2f}" for i in top), flush=True)
        pos, neg = X[y > 0], X[y == 0]
        print(f"    mean at positives: " + " ".join(f"{NAMES[i]}={np.nanmean(pos[:, i]):.3f}" for i in range(12)), flush=True)
        print(f"    mean at negatives: " + " ".join(f"{NAMES[i]}={np.nanmean(neg[:, i]):.3f}" for i in range(12)), flush=True)
    for name, mdl in M.items():
        ious, aucs = [], []
        for x, y, e in zip(*R["TEST"][0][:3]):
            pp = mdl.predict(x, num_threads=8); s = best_set(np.where(e, pp, 0.0), e); u = (s | y) & e
            ious.append(1.0 if not u.any() else (s & y & e).sum() / u.sum())
            yy, q = y[e], pp[e]
            if 0 < yy.sum() < len(yy):
                rk = rankdata(q); aucs.append((rk[yy].sum() - yy.sum() * (yy.sum() + 1) / 2) / (yy.sum() * (len(yy) - yy.sum())))
        print(f"  test (real 6-8) with {name}: IoU {np.mean(ious):.3f}, AUC {np.mean(aucs):.3f}", flush=True)
    Xt, yt = stackd(R["TEST"])
    print(f"    real TEST mean at positives: " + " ".join(f"{NAMES[i]}={np.nanmean(Xt[yt > 0][:, i]):.3f}" for i in range(12)), flush=True)
