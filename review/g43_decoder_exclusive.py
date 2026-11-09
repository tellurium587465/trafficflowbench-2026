"""G43: dependence-aware expected-IoU decoder for onset (literature D, rank 1). best_set maximises E[|S∩Y|]/E[|S∪Y|]
(ratio of expectations). When the true queue sits at ONE of several candidate sites (mutually exclusive: one
bottleneck breaks first), the ratio approximation over-selects: with two equal sites and P(A)=p the optimum is A alone
for p > 1/2, but the approximation keeps A∪B until p < 2/3 (up to 0.167 IoU lost per window).
Model for the T+30 vector p (onset predicts only T+30):
  components c = runs of links with p >= 0.02, merged across gaps <= 2 links;
  the site is ONE component, P(c) = m_c / sum m (m_c = max p in c);
  given c, link i in c is queued with probability min(1, p_i / m_c), independently (the top link of c for sure).
Candidates: for each component, its cells ranked by p, every prefix length; plus the union of the best prefixes of the two
heaviest components. The candidate with the highest Monte-Carlo mean IoU (400 draws) is chosen. Compared with best_set
on the month-CV OOF of the no-memory NN (morning onset windows), by number of components the current decoder selects.
usage: python g43_decoder_exclusive.py"""
import sys, numpy as np, pandas as pd
sys.path.insert(0, r"C:/Users/user/tfb/src")
from t2_eval_oof import T, load_oof
from t2_model import best_set
SPEC = "q25_nomem_s3+q25_nomem_s4+q31_nomem_wp05_s0:2+q31_nomem_attn1_s0:2+q31_nomem_r13_s0+q31_nomem_drop1_s0"
oof = load_oof(SPEC); rng = np.random.default_rng(0)


def components(p, thr=0.02, gap=int(__import__("os").environ.get("G43_GAP", "2"))):
    idx = np.flatnonzero(p >= thr); comps = []
    for i in idx:
        if comps and i - comps[-1][-1] <= gap + 1:
            comps[-1].append(i)
        else:
            comps.append([i])
    return [np.array(c) for c in comps]


def exclusive_decode(p, draws=400):
    comps = components(p)
    if len(comps) <= 1:
        return None                                              # nothing to do: same as best_set
    m = np.array([p[c].max() for c in comps]); pc = m / m.sum()
    # Monte-Carlo label sets
    Ys = np.zeros((draws, p.size), bool); site = rng.choice(len(comps), draws, p=pc)
    for d in range(draws):
        c = comps[site[d]]; Ys[d, c] = rng.random(len(c)) < np.minimum(1.0, p[c] / m[site[d]])
    def mean_iou(S):
        inter = (Ys & S).sum(1); uni = (Ys | S).sum(1)
        return np.where(uni == 0, 1.0, inter / np.maximum(uni, 1)).mean()
    best, bestS = -1, None; prefix_best = []
    for c in comps:
        order = c[np.argsort(-p[c])]; cb, cS = -1, None
        for k in range(1, len(order) + 1):
            S = np.zeros(p.size, bool); S[order[:k]] = True; v = mean_iou(S)
            if v > cb:
                cb, cS = v, S
        prefix_best.append((cb, cS))
        if cb > best:
            best, bestS = cb, cS
    top2 = np.argsort(-m)[:2]; S = prefix_best[top2[0]][1] | prefix_best[top2[1]][1]; v = mean_iou(S)
    if v > best:
        best, bestS = v, S
    return bestS


rows = []
for k in oof:
    if not T.META["is_sel"][k] or T.META["cond"][k] != "queue_onset" or (int(T.META["t"][k]) + 6) % 288 >= 144:
        continue
    lab, el = T.LAB[k]; y = (lab[-1] & el[-1]).astype(bool); e = el[-1].astype(bool)
    p = oof[k][-1].copy()
    q = np.zeros_like(oof[k]); q[-1] = p
    s0 = best_set(q.ravel(), np.ones(q.size, bool)).reshape(q.shape)[-1]
    s1 = exclusive_decode(p); s1 = s0 if s1 is None else s1
    def iou(s):
        u = ((s | y) & e).sum(); return 1.0 if u == 0 else (s & y & e).sum() / u
    nsel = len(components(s0.astype(float), thr=0.5))
    rows.append(dict(cl=str(T.META["panel"][k]) + str(T.META["date"][k])[:10], ncomp_sel=nsel, ncomp_p=len(components(p)),
                     cur=iou(s0), excl=iou(s1), changed=bool((s0 != s1).any())))
R = pd.DataFrame(rows); d = (R.excl - R.cur).to_numpy(); u = R.cl.unique(); idx = {x: np.flatnonzero(R.cl.to_numpy() == x) for x in u}
se = np.std([d[np.concatenate([idx[x] for x in rng.choice(u, len(u))])].mean() for _ in range(1000)])
print(f"morning onset windows {len(R)}; current decoder selects >= 2 separate sites in {np.mean(R.ncomp_sel >= 2):.3f}; p has >= 2 components in {np.mean(R.ncomp_p >= 2):.3f}; set changed in {R.changed.mean():.3f}")
print(f"IoU current {R.cur.mean():.4f}, exclusive {R.excl.mean():.4f}, diff {d.mean():+.4f} (SE {se:.4f})")
print(R.groupby(R.ncomp_sel.clip(upper=3))[["cur", "excl"]].agg(["mean", "size"]).round(4).to_string())
