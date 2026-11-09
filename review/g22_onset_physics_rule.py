"""G22: how predictable is the onset location from the current state alone, with no learning? On the onset windows:
rank links by the speed ratio v / v_cut at the last history row (noisy observation, eligible), predict the k lowest
(k = 1, 3, 5) at T+30, plus the variant 'lowest link +-1'. IoU at T+30 against the label; compared with the month-CV
and LOPO NN OOF (expected-IoU set). If the rule is far above LOPO, the LOPO path is broken or the NN learned no
transportable onset physics."""
import sys, numpy as np, pandas as pd
sys.path.insert(0, r"C:/Users/user/tfb/src")
import t2_eval_surprise as S
from t2_eval_oof import T, oof_of
from cachelib import load

A = oof_of("q25_nomem_s3"); B = oof_of("q33L_nomem_s3"); PC = {}


def bestset(p):
    pe = p.ravel(); o = np.argsort(-pe); cs = np.cumsum(pe[o]); kk = np.arange(1, pe.size + 1)
    kb = int(np.argmax(cs / (kk + pe.sum() - cs))) + 1; s = np.zeros(pe.size, bool); s[o[:kb]] = True
    return s.reshape(p.shape)


def iou(s, lab, el):
    u = ((s | lab) & el).sum()
    return 1.0 if u == 0 else (s & lab & el).sum() / u


R = []
for k in S.SEL:
    p = str(T.META["panel"][k]); t = int(T.META["t"][k])
    if p not in PC:
        PC[p] = load(p, "train")
    P = PC[p]; lab, el = T.LAB[k]; lab = lab.astype(bool); el = el.astype(bool)
    r = P.V[t - 1] / P.vcut; r = np.where(np.isfinite(r), r, np.inf)
    o = np.argsort(r); row = {}
    for kk in (1, 3, 5):
        s = np.zeros_like(lab); s[-1, o[:kk]] = True; row[f"low{kk}"] = iou(s, lab, el)
    s = np.zeros_like(lab); j = o[0]; s[-1, max(j - 1, 0):j + 2] = True; row["low1+-1"] = iou(s, lab, el)
    row["nn_month"] = iou(bestset(A[k]), lab, el); row["nn_lopo"] = iou(bestset(B[k]), lab, el)
    row["size"] = lab[-1].sum(); row["minratio"] = r[o[0]]
    R.append(row)
R = pd.DataFrame(R)
print(R.mean().round(4).to_string())
