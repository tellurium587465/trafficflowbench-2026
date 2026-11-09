"""Review: onset as a choice among each panel's top candidate sites. Accuracy by the true site's frequency rank,
and whether a location-free physics score picks the right candidate among the top sites."""
import os, sys, numpy as np
from collections import defaultdict, Counter
sys.path.insert(0, r"C:/Users/user/tfb/src")
os.environ.setdefault("TFB_T2CACHE", "2_noT")
import t2_eval_oof as E
from t2_model import best_set, FOLDS

M = E.T.META; LAB = E.T.LAB; cond = M["cond"]; panel = M["panel"]
oL = E.oof_of("l127_fast_af"); oN = E.oof_of("q12_af_r13_nopos")
keys = sorted(set(oL) & set(oN)); PB = {k: 0.5 * oL[k] + 0.5 * oN[k] for k in keys}
mon = np.array([d[:7] for d in M["date"]]); fold_of = {m: i for i, f in enumerate(FOLDS) for m in f}
allon = [k for k in range(E.T.NWIN) if cond[k] == "queue_onset"]
R = 3
sites = {}
for f in range(3):
    acc = {}
    for k in allon:
        if fold_of.get(mon[k], -1) == f: continue
        a = acc.setdefault(panel[k], [0, 0]); a[0] = a[0] + LAB[k][0][5].astype(float); a[1] += 1
    for p, (s, n) in acc.items():
        fr = np.convolve(s / n, np.ones(2 * R + 1), "same"); lst = []
        for _ in range(3):
            j = int(np.argmax(fr)); lst.append((j, float(fr[j]))); fr[max(0, j - 2 * R):j + 2 * R + 1] = -1
        sites[(f, p)] = lst


def site_of(f, p, j):
    for r, (c, _) in enumerate(sites[(f, p)]):
        if abs(j - c) <= R: return r
    return 3                                    # outside the top-3


d = "E:/tfb_nn/nn_data_afdem/"
nm = list(np.load(d + "names.npy", allow_pickle=True)); X = np.load(d + "X.npy", mmap_mode="r"); OFF = np.load(d + "off.npy")
phys = ["dem30_over_dnmin", "arr6", "dem30_c", "q_m3", "v_m3"]; ci = [nm.index(c) for c in phys]
iou = lambda s, lab, el: 1.0 if ((s | lab) & el).sum() == 0 else ((s & lab & el).sum()) / ((s | lab) & el).sum()
tab = defaultdict(list); conf = Counter(); hacc = defaultdict(lambda: defaultdict(list))
for k in keys:
    if cond[k] != "queue_onset" or not LAB[k][0][5].any(): continue
    f = fold_of[mon[k]]; p = panel[k]; lab = LAB[k][0][5]
    q = np.flatnonzero(lab); ts = Counter(site_of(f, p, j) for j in q).most_common(1)[0][0]
    for nmod, P in (("blend", PB), ("LGBM", oL), ("NN", oN)):
        ms = site_of(f, p, int(np.argmax(P[k][5])))
        tab[(nmod, ts)].append((ms == ts, iou(best_set(P[k].ravel(), np.ones(P[k].size, bool)).reshape(P[k].shape), LAB[k][0], LAB[k][1])))
        if nmod == "blend": conf[(ts, ms)] += 1
    if ts < 3:
        Xi = np.asarray(X[OFF[k]:OFF[k + 1]])[:, ci]
        for j, c in enumerate(phys):
            v = Xi[:, j].astype(float); sc = []
            for r, (cc, _) in enumerate(sites[(f, p)]):
                seg = v[max(0, cc - R):cc + R + 1]; seg = seg[np.isfinite(seg)]
                sc.append((-seg.min() if c == "v_m3" else seg.max()) if len(seg) else -np.inf)
            hacc[c][ts].append(int(np.argmax(sc)) == ts)
        hacc["always_rank1"][ts].append(ts == 0)
        ms = site_of(f, p, int(np.argmax(PB[k][5]))); hacc["blend(model)"][ts].append(ms == ts)
n_by = Counter(ts for (m, ts) in tab if m == "blend" for _ in tab[(m, ts)])
print("true-site share (sel+eval onset):", {r: len(tab[("blend", r)]) for r in range(4)})
for m in ("blend", "LGBM", "NN"):
    print(m, " | ".join(f"rank{r+1 if r<3 else '-other'}: site-acc {np.mean([a for a,_ in tab[(m,r)]]):.3f} IoU {np.mean([b for _,b in tab[(m,r)]]):.3f}" for r in range(4)))
print("blend confusion (true rank -> chosen rank):")
for t in range(4):
    print("  ", t, [conf[(t, m)] for m in range(4)])
print("choice among top-3 candidates only (accuracy by true rank 1/2/3):")
for c, dd in hacc.items():
    print(f"  {c:18s}", " ".join(f"{np.mean(dd[r]):.3f}(n={len(dd[r])})" for r in range(3)))
