"""G104 (10/24, X lane 2, LITERATURE_SCAN_20261024 §2-1): MBR decoding of the onset T+30 set on joint samples built from the
final A's own probabilities (calibrated, G102). G103's flow-hazard sampler is far more diffuse than reality (its realized IoU
0.49-0.59 exceeds its own expectation 0.25-0.29), so the joint model here keeps the decoder's marginals pb exactly and only
adds dependence:
  blocks = contiguous runs of T+30 links with pb > 0.02; within a block comonotone (one uniform per block: cell j queued
  iff U_b < pb_j -> a queue grows from its most likely cells); between blocks independent (pre-declared, k = 1;
  G43's exclusive coupling was -0.013).
Candidates: for every block the nested sets {cells with pb >= level} (levels = the block's distinct pb values, top 8) and the
empty set, combined over the 3 heaviest blocks (others left out) -> the set with the largest expected IoU over 4000 samples.
Columns (per window, k expectation, evidence rule "cur" as production): A = production decoder (best_set over the 6 x n p),
bs5 = best_set on the T+30 row only (onset is scored at T+30 only), mbr = the MBR set. Writes work/t2/g104_onset_mbr.csv.
usage: TFB_M3VER=v3 python g104_onset_mbr.py"""
import os, sys, pickle, itertools
os.environ.setdefault("TFB_M3VER", "v3")
sys.path.insert(0, r"C:/Users/user/tfb/src"); argv0 = sys.argv[0]
import numpy as np, pandas as pd, lightgbm as lgb
import t2_phys_onset as PO
sys.argv = [argv0, "0,1,2,3"]
import t2_m3_gate as G, t2_m3v2_gate as V, m3v2
from t2_model import best_set
from common import NN

W = G.W; VER = G.VER; PRE = G.PRE
CF5 = (("attnglob", 2.0), ("attnwp05", 2.0), ("attn1", 2.0), ("drop1", 1.0), ("r13", 1.0))
ON = [(f"q24f_{c}_s{s}", w) for s in range(8) for c, w in CF5]; PRO = {t: pickle.load(open(W + f"probs_{PRE}__{t}.pkl", "rb")) for t, _ in ON}
X = np.load(f"{NN}/nn_test_m3{VER}/X.npy", mmap_mode="r"); OFF = np.load(f"{NN}/nn_test_m3{VER}/off.npy")
phys = lgb.Booster(model_file=W + "physon_base.txt"); ev = PO.month_events("validation", PO.rarity_table())
rng = np.random.default_rng(104); NS = 4000


def mbr(p5):
    on = np.r_[False, p5 > 0.02, False]; d = np.diff(on.astype(int)); st, en = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
    blocks = sorted([np.arange(a, b) for a, b in zip(st, en)], key=lambda c: -p5[c].sum())[:3]
    if not blocks:
        return np.zeros(len(p5), bool), 1.0
    Ys = []                                                      # an onset window has a queue at T+6 by construction:
    for _ in range(50):                                          # condition the samples on a non-empty set
        U = rng.random((NS, len(blocks))); Y = np.zeros((NS, len(p5)), bool)
        for b, c in enumerate(blocks):
            Y[:, c] = U[:, b:b + 1] < p5[c][None, :]
        Ys.append(Y[Y.any(1)])
        if sum(len(y) for y in Ys) >= NS // 2:
            break
    Y = np.vstack(Ys)
    if len(Y) == 0:
        return best_set(p5, np.ones(len(p5), bool)), 0.0
    opts = []
    for c in blocks:
        lv = np.unique(p5[c])[::-1][:8]
        opts.append([np.zeros(0, int)] + [c[p5[c] >= l] for l in lv])
    best, be = None, -1.0
    for combo in itertools.product(*opts):
        s = np.concatenate(combo); i = Y[:, s].sum(1); u = Y.sum(1) + len(s) - i
        e = float(np.where(u == 0, 1.0, i / np.maximum(u, 1)).mean())
        if e > be:
            best, be = s, e
    sel = np.zeros(len(p5), bool); sel[best] = True
    return sel, be


cols = {v: {} for v in ("A", "bs5", "mbr", "bsc")}; ex = []
for i, w in enumerate(G.WID):
    m = G.META[w]
    if m["cond"] != "queue_onset" or m3v2.is_holdout(np.array([m["day"]]))[0]:   # the hold-out is not computed
        continue
    Xw = np.asarray(X[OFF[i]:OFF[i + 1]]); p = sum(wt * PRO[t][w] for t, wt in ON) / sum(wt for _, wt in ON); a = PO.adaptive_a(ev, m["day"])
    p = p.copy(); p[5] = (1 - a) * p[5] + a * phys.predict(PO.feats(Xw))
    for v in cols:
        cols[v][w] = 0.0
    for k, pr in m3v2.kdist(m["day"]).items():
        pb = p.copy(); s, nd = m3v2.seen(m["panel"], m["t"], k, -6, 12)
        if nd:
            pb[-1] = pb[-1] * np.where(G.dil(s, 2), 1.0, 0.25)
        sel = best_set(pb.ravel(), np.ones(pb.size, bool)).reshape(pb.shape); assert (sel == V.onset_pred(p, m, "cur", k)).all()
        cols["A"][w] += pr * G.iou(sel, m)
        s5 = np.zeros_like(sel); s5[-1] = best_set(pb[-1], np.ones(pb.shape[1], bool)); cols["bs5"][w] += pr * G.iou(s5, m)
        sm = np.zeros_like(sel); sm[-1], e = mbr(pb[-1]); cols["mbr"][w] += pr * G.iou(sm, m); ex.append(e)
        pc = pb.copy(); pc[-1] = np.minimum(1.0, pb[-1] / max(1 - np.prod(1 - np.clip(pb[-1], 0, 1)), 1e-9))   # P(j | a queue at T+6), independent cells
        cols["bsc"][w] += pr * G.iou(best_set(pc.ravel(), np.ones(pc.size, bool)).reshape(pc.shape), m)
D = V.frame(cols); D.to_csv(W + "g104_onset_mbr.csv", index=False)
dev = ~m3v2.is_holdout(D.day)
print(f"M3 {VER} dev onset windows {int((dev & (D.cond == 'queue_onset')).sum())}: A {D.A[dev].mean():.4f} | bs5 {D.bs5[dev].mean():.4f} | mbr {D.mbr[dev].mean():.4f} "
      f"| MBR expected (model) {np.mean(ex):.4f}")
