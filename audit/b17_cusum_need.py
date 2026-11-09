"""B17 (10/29, strategy lane 7 item 2-7, need test before building): do CUSUM / change-point precursor statistics of the
raw last hour add to the physics onset ranker's inputs? Training-month onset windows with a T+30 queue (the ranker's rows).
Per link, from the window's own rows t-12..t-1 (released values, NaN skipped): one-sided CUSUM of q / cap above its
first-half mean (slack 0.01) and of the speed drop (vf - v) / vf (slack 0.01), the end value and the maximum, raw and as
within-window percentiles (8 features). LightGBM with the ranker's parameters, 9 folds by month: pooled AUC and per-window
top-1 hit, base (t2_phys_onset.feats) vs base + CUSUM.
Pre-declared reading: AUC gain < 0.002 and top-1 gain < 0.01 -> close (X3 already found no signal in the raw history beyond
the summaries); otherwise build into the ranker and judge on M3 v3 dev onset (house rule)."""
import sys, numpy as np, lightgbm as lgb
sys.path.insert(0, r"C:/Users/user/tfb/src")
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
from cachelib import load
from common import NN
import t2_phys_onset as PO

d = f"{NN}/nn_data_afdem"
X = np.load(d + "/X.npy", mmap_mode="r"); Y = np.load(d + "/Y.npy", mmap_mode="r"); E = np.load(d + "/EL.npy", mmap_mode="r")
off = np.load(d + "/off.npy"); cond = np.load(d + "/cond.npy", allow_pickle=True); pan = np.load(d + "/panel.npy", allow_pickle=True)
tt = np.load(d + "/t.npy"); dt = np.load(d + "/date.npy", allow_pickle=True)


def pct(v):
    out = np.full_like(v, 0.5); ok = np.isfinite(v)
    if ok.sum() >= 2 and np.nanstd(v) > 0:
        out[ok] = (rankdata(v[ok]) - 0.5) / ok.sum()
    return out


def cusum(H, slack=0.01):
    """H (12, n) with NaN; one-sided CUSUM above the first-half mean -> (end, max)"""
    mu = np.nanmean(H[:6], 0); S = np.zeros(H.shape[1]); M = np.zeros(H.shape[1])
    for r in range(H.shape[0]):
        x = H[r] - mu - slack; S = np.where(np.isfinite(x), np.maximum(0, S + x), S); M = np.maximum(M, S)
    return S, M


B, C, Ys, G, MO = [], [], [], [], []
for p in np.unique(pan):
    P = load(str(p), "train")
    for k in np.flatnonzero((cond == "queue_onset") & (pan == p)):
        a, b = off[k], off[k + 1]; y = np.asarray(Y[a:b, 5]).astype(bool); e = np.asarray(E[a:b, 5]).astype(bool)
        if not (y & e).any():
            continue
        t = int(tt[k]); Hq = P.Q[t - 12:t] / P.cap[None, :]; Hv = (P.vf[None, :] - P.V[t - 12:t]) / P.vf[None, :]
        with np.errstate(all="ignore"):
            sq, mq = cusum(Hq); sv, mv = cusum(Hv)
        F = np.column_stack([sq, mq, sv, mv, pct(sq), pct(mq), pct(sv), pct(mv)])
        B.append(PO.feats(np.asarray(X[a:b]))[e]); C.append(F[e]); Ys.append(y[e].astype(int)); G.append(np.full(e.sum(), k)); MO.append(np.full(e.sum(), str(dt[k])[:7]))
    load.cache_clear()
B, C, Ys, G, MO = map(np.concatenate, (B, C, Ys, G, MO)) if False else (np.vstack(B), np.vstack(C), np.concatenate(Ys), np.concatenate(G), np.concatenate(MO))
print("rows", len(Ys), "windows", len(np.unique(G)), "positives", int(Ys.sum()), flush=True)
res = {}
for nm, F in (("base", B), ("base+cusum", np.column_stack([B, C]))):
    pr = np.zeros(len(Ys))
    for m in np.unique(MO):
        tr, te = MO != m, MO == m
        pr[te] = lgb.train(PO.PRM, lgb.Dataset(F[tr], Ys[tr]), 400).predict(F[te])
    top = [Ys[G == g][np.argmax(pr[G == g])] for g in np.unique(G)]
    res[nm] = (roc_auc_score(Ys, pr), float(np.mean(top)))
    print(f"{nm:12s} AUC {res[nm][0]:.4f} | top-1 hit {res[nm][1]:.4f}", flush=True)
print(f"gain: AUC {res['base+cusum'][0] - res['base'][0]:+.4f}, top-1 {res['base+cusum'][1] - res['base'][1]:+.4f}")
