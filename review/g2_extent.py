"""Gem checks on onset windows (selector-like train set, nomem NN OOF and the s31-like blend):
(a) ranking vs extent: IoU of the expected-IoU set vs the top-m set with the true m (oracle size) -> how much is lost by
    choosing the set size; (b) structured decoding: keep only the largest connected block of links at T+30."""
import sys, numpy as np
sys.path.insert(0, r"C:/Users/user/tfb/src")
import t2_eval_surprise as S
from t2_eval_oof import T, load_oof


def iou(s, lab, el):
    u = ((s | lab) & el).sum()
    return 1.0 if u == 0 else (s & lab & el).sum() / u


def bestset(p):
    pe = p.ravel(); o = np.argsort(-pe); cs = np.cumsum(pe[o]); kk = np.arange(1, pe.size + 1)
    kb = int(np.argmax(cs / (kk + pe.sum() - cs))) + 1; s = np.zeros(pe.size, bool); s[o[:kb]] = True
    return s.reshape(p.shape)


def largest_block(row):
    idx = np.flatnonzero(row)
    if len(idx) == 0:
        return row
    runs = np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1); best = max(runs, key=len)
    out = np.zeros_like(row); out[best] = True
    return out


for name, spec in (("nomem NN", "q23_nomem_s3+q23_nomem_s4"),
                   ("s31-like blend", "l127_full_af:2+q14_attn1_glob+q16_attnglob_s1+q14_attn1_wp05:2+q13_attn1:2+q12_af_r13_nopos+q17_drop1")):
    O = load_oof(spec); R = []
    for j, k in enumerate(S.SEL):
        if k not in O:
            continue
        lab, el = T.LAB[k]; lab = lab.astype(bool); el = el.astype(bool); p = O[k]
        s = bestset(p); m = int((lab & el).sum())
        pe = np.where(el, p, -1.0).ravel()                     # oracle size: top-m among scored (eligible) cells only
        o = np.argsort(-pe); so = np.zeros(p.size, bool); so[o[:max(m, 1)]] = True; so = so.reshape(p.shape)
        sb = s.copy(); sb[-1] = largest_block(s[-1] & el[-1])
        nb = len(np.split(np.flatnonzero(lab[-1] & el[-1]), np.flatnonzero(np.diff(np.flatnonzero(lab[-1] & el[-1])) > 1) + 1)) if (lab[-1] & el[-1]).any() else 0
        R.append((S.F[j] < 0.3, iou(s, lab, el), iou(so, lab, el), iou(sb, lab, el), (s & el).sum(), m, nb, (lab & el)[:-1].sum()))
    R = np.array(R, float)
    for lab_, msk in (("all", np.ones(len(R), bool)), ("rare f<0.3", R[:, 0] == 1)):
        r = R[msk]
        print(f"{name:15s} {lab_:10s} n={msk.sum():4d}  IoU best_set {r[:,1].mean():.4f}  oracle size {r[:,2].mean():.4f}  "
              f"largest block {r[:,3].mean():.4f}  | mean |S| {r[:,4].mean():.2f} vs true {r[:,5].mean():.2f}  "
              f"over {np.mean(r[:,4] > r[:,5]):.2f} under {np.mean(r[:,4] < r[:,5]):.2f} | true blocks at T+30 >1: {np.mean(r[:,6] > 1):.2f}, true cells before T+30 {r[:,7].mean():.2f}")
