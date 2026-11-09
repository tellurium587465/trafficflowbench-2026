"""(1) Label-free bounds for LB comparisons: |IoU(A,Y)-IoU(B,Y)| <= 1-IoU(A,B) (Jaccard distance is a metric).
    -> max possible |dS_queue| and a null SE (random sign) per pair, on validation and private windows.
(2) Mechanism check: does the per-link free-flow offset of a test month (vs train) explain where congestion moved?
Read-only; stdout."""
import sys, os, glob, pickle, numpy as np, pandas as pd
sys.path.insert(0, r"C:/Users/user/tfb/src"); os.chdir(r"C:/Users/user/tfb/src")
from cachelib import load, PANELS, T2_PANELS

wi = pd.concat([pd.read_csv(f) for f in glob.glob(r"C:/Users/user/tfb/data/kaggle_public/task2/*/*/window_index.csv")])
cond = dict(zip(wi.window_id, wi.condition))
W = r"C:/Users/user/tfb/work/t2/"
M = {"v8": "probs_v8.pkl", "v9npp": "probs_v9npp.pkl", "s17": "probs_s17_t1f3ens3c_t2v12nppdem_nnDnp6_t4l2.pkl",
     "s17L": "probs_s17L_t1f3ens3c_t2v12nppdem_t4l2.pkl", "s23": "probs_s23_t1GapML_t2v14af_nnAf6_t4l2.pkl",
     "s16": "probs_s16_t1fdv075_t2lgbm3nnmix9_t4l2.pkl"}
PR = {m: pickle.load(open(W + f, "rb")) for m, f in M.items()}
def bs(o):
    pe = np.asarray(o).ravel(); order = np.argsort(-pe); cs = np.cumsum(pe[order]); kk = np.arange(1, pe.size + 1)
    kb = int(np.argmax(cs / (kk + pe.sum() - cs))) + 1; s = np.zeros(pe.size, bool); s[order[:kb]] = True; return s
print("# (1) Jaccard-distance bound per pair: mean d (= max |dS| for that condition), null SE sqrt(sum d^2)/n, #windows d>0.5")
for a, c in (("v8", "v9npp"), ("v9npp", "s17"), ("s17L", "s17"), ("s17", "s23"), ("s16", "s17")):
    for sp in ("validation", "private"):
        res = []
        for cd in ("queue_onset", "queue_ongoing"):
            ws = [w for w in PR[a] if f"_{sp}_" in w and cond[w] == cd]
            d = np.array([1 - (bs(PR[a][w]) & bs(PR[c][w])).sum() / max((bs(PR[a][w]) | bs(PR[c][w])).sum(), 1) for w in ws])
            res.append(f"{cd[6:]:7s} max {d.mean():.3f} nullSE {np.sqrt((d**2).sum())/len(d):.3f} n>0.5 {int((d>0.5).sum()):2d}")
        dall = None
        print(f"{a:5s}->{c:5s} {sp:10s} " + " | ".join(res))
print("S_queue = (onset+ongoing)/2 ; total = 0.30*S_queue")

print("\n# (2) per-link free-flow offset (test month vs train) vs change of congestion rate at that link")
rows = []
for p in T2_PANELS:
    tr = load(p, "train")
    months = np.array([t[:7] for t in tr.times])
    def ff(P, r=None):
        V = P.V if r is None else P.V[r]; ok = np.isfinite(V) & (V >= 0.85 * P.vf[None, :])
        return np.nanmedian(np.where(ok, V, np.nan), axis=0)
    def qr(P, r=None):
        V = P.V if r is None else P.V[r]; E = P.E if r is None else P.E[r]; obs = np.isfinite(V) & E
        return ((np.nan_to_num(V, nan=1e9) <= P.vcut[None, :]) & obs).sum(0) / np.maximum(obs.sum(0), 1)
    base_ff = np.nanmean([ff(tr, months == m) for m in sorted(set(months))], 0); base_q = qr(tr)
    for sp in ("validation", "private"):
        P = load(p, sp); d = ff(P) / base_ff - 1; q = qr(P)
        hot = base_q > 0.01                                     # links that congest in train (>1% of rows)
        dq = q - base_q
        # downstream-neighbour offset (bottleneck capacity is set by the link where the queue head sits)
        r_all = pd.Series(d).corr(pd.Series(dq), method="spearman")
        r_hot = pd.Series(d[hot]).corr(pd.Series(dq[hot]), method="spearman") if hot.sum() > 4 else np.nan
        # head share: congestion-weighted mean offset of the links that gained vs lost congestion
        gain = dq > 0.005; lose = dq < -0.005
        rows.append(dict(panel=p, split=sp, n_hot=int(hot.sum()), rho_all=r_all, rho_hot=r_hot,
                         d_gain=np.nanmean(d[gain]) * 100 if gain.any() else np.nan,
                         d_lose=np.nanmean(d[lose]) * 100 if lose.any() else np.nan))
        del P
    del tr
r = pd.DataFrame(rows)
print(r.round(3).to_string(index=False))
print(r.groupby("split")[["rho_all", "rho_hot", "d_gain", "d_lose"]].mean().round(3).to_string())
