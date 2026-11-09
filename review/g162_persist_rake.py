"""G162 (11/01, ledger #116 addendum, a robustness check of G156's "not identified"): the mean backward persistence differs (M3 dev 0.86, official 0.77-0.82) while the 6-feature logistic has
AUC near 0.5. Each history feature alone: AUC official vs M3 dev, and the level of final A's March flow after raking M3 dev to the official share in the feature's quintile bins
(bins from the pooled values, on top of the March composition weights, cluster bootstrap). Reads work/t2/g156_feats.csv and g156_m3_frame.csv.
usage: TFB_M3VER=v3 py -B -X utf8 g162_persist_rake.py"""
import sys
sys.dont_write_bytecode = True; sys.path.insert(0, r"C:/Users/user/tfb/src"); sys.argv = [sys.argv[0], "0"]
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
import m3v2

W = "C:/Users/user/tfb/work/t2/"; F = pd.read_csv(W + "g156_feats.csv"); D = pd.read_csv(W + "g156_m3_frame.csv"); M = F[F.src == "M3"].set_index("w").loc[D.w].reset_index(); O = F[F.src != "M3"]; w0 = m3v2.weights(D, "validation")[0]; A = D.A.to_numpy()
cl = pd.factorize(D.panel + "_" + D.day.astype(str))[0]; n = cl.max() + 1; Rm = np.random.default_rng(0).multinomial(n, np.full(n, 1 / n), size=2000)


def move(w1):
    f = lambda w: (Rm @ np.bincount(cl, A * w, n)) / (Rm @ np.bincount(cl, w, n)); return float((A * w1).sum() / w1.sum()), float((f(w1) - f(w0)).std())


print(f"base level {(A * w0).sum() / w0.sum():.4f}")
print("A by persist quintile on M3 dev:", D.A.groupby(pd.qcut(M.persist, 5, duplicates="drop").astype(str).to_numpy()).mean().round(3).to_dict())
for sel, tag in ((("March", "April", "June"), "all official"), (("March",), "March")):
    o = O[O.src.isin(sel)]
    for c in ("share", "last", "growth", "persist", "run", "age"):
        auc = roc_auc_score(np.r_[np.zeros(len(M)), np.ones(len(o))], np.r_[M[c], o[c]]); e = np.unique(np.quantile(np.r_[M[c], o[c]], [.2, .4, .6, .8])); bm = np.digitize(M[c], e); bo = np.digitize(o[c], e)
        tgt = np.bincount(bo, minlength=len(e) + 1) / len(o); w1 = w0.copy()
        for b in range(len(e) + 1):
            if (bm == b).any():
                w1[bm == b] *= tgt[b] / max(w0[bm == b].sum(), 1e-9)
        w1 /= w1.sum(); lv, sm = move(w1)
        print(f"{tag:12s} {c:8s} AUC {max(auc, 1 - auc):.3f} | official mean {o[c].mean():.3f} M3 {M[c].mean():.3f} | level {lv:.4f} move {lv - (A * w0).sum() / w0.sum():+.4f} (SE {sm:.4f}) ESS {1 / np.sum(w1 ** 2):.0f}")
