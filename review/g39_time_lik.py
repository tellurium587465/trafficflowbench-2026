"""G39: onset post-processing with a site-time likelihood (G38: the breakdown time identifies the site, prior 0.819 ->
posterior 0.870 on months 6-8). For an onset window at T, links near a regular site s (head bucket within +-2 links,
G34 events of the OTHER months) get their T+30 probability multiplied by (L_s(T+30) / Lbar)^gamma, where L_s is the
normal density of the site's first-queue slot and Lbar the mean over the panel's sites; links far from any site keep
theirs. OOF of the no-memory NN on the selector-like windows (as G23 / G30), plain / weighted / f<0.3, nested gamma.
usage: python g39_time_lik.py"""
import sys, numpy as np, pandas as pd
from scipy.stats import norm
sys.path.insert(0, r"C:/Users/user/tfb/src")
import t2_eval_surprise as S
from t2_eval_oof import T, load_oof
from cachelib import load
SPEC = "q25_nomem_s3+q25_nomem_s4+q31_nomem_wp05_s0:2+q31_nomem_attn1_s0:2+q31_nomem_r13_s0+q31_nomem_drop1_s0"
oof = load_oof(SPEC); E = pd.read_csv(r"C:/Users/user/tfb/work/t2/g34_events.csv")
MON = {}
GAM = (0.0, 0.25, 0.5, 1.0)
FOLD = {**{m: 0 for m in ("2030-06", "2030-07", "2030-08")}, **{m: 1 for m in ("2030-09", "2030-10", "2030-11")},
        **{m: 2 for m in ("2030-12", "2031-01", "2031-02")}}


def sites(p, month):
    key = (p, month)
    if key not in MON:
        P = load(p, "train"); g = E[E.panel == p]; mon = np.array([str(P.day[d])[:7] for d in g.day])
        g = g[mon != month]; out = []
        for b, h in g.groupby("b"):
            if len(h) >= 3:
                out.append((int(b), h.slot.mean(), max(h.slot.std() if len(h) > 1 else 12, 3)))
        MON[key] = out
    return MON[key]


def iou_last(pl, base, lab, el):
    p = base.copy(); p[-1] = pl
    pe = p.ravel(); o = np.argsort(-pe); cs = np.cumsum(pe[o]); kk = np.arange(1, pe.size + 1)
    kb = int(np.argmax(cs / (kk + pe.sum() - cs))) + 1; s = np.zeros(pe.size, bool); s[o[:kb]] = True; s = s.reshape(p.shape)
    u = ((s | lab) & el).sum(); return 1.0 if u == 0 else (s & lab & el).sum() / u


M = np.zeros((len(GAM), len(S.SEL))); fold = np.zeros(len(S.SEL), int); keepw = np.zeros(len(S.SEL), bool)
for i, k in enumerate(S.SEL):
    p = str(T.META["panel"][k]); t = int(T.META["t"][k]); mon = str(T.META["date"][k])[:7]; fold[i] = FOLD[mon]
    lab, el = T.LAB[k]; lab = lab.astype(bool); el = el.astype(bool); b0 = oof[k]; n = b0.shape[1]
    st = sites(p, mon); slot = (t + 6) % 288
    keepw[i] = slot < 144                                         # review 10/13: official onset windows are mornings
    rel = np.ones(n)
    inside = [(b, m, s) for (b, m, s) in st if abs(slot - m) <= 3 * s]   # weights only inside the sites' time support
    if inside:
        L = np.array([norm.pdf(slot, m, s) for (_, m, s) in inside]); Lbar = L.mean()
        best = np.full(n, -1.0)
        for (b, m, s), l in zip(inside, L):
            lo, hi = max(0, b - 6), min(n, b + 1)                    # the queue sits upstream of the head b
            best[lo:hi] = np.maximum(best[lo:hi], l / Lbar)
        rel = np.where(best >= 0, best, 1.0)
    for j, g in enumerate(GAM):
        M[j, i] = iou_last(np.clip(b0[-1] * np.clip(rel, 1e-3, 1e3) ** g, 0, 1), b0, lab, el)
M[:, ~keepw] = np.nan                                            # morning windows only (boot_se skips NaN)
W1, SUB = S.W1, S.SUB; a = np.flatnonzero(keepw)
print(f"morning onset windows {keepw.sum()} of {len(keepw)}")
wm = lambda r, idx: np.sum(W1[idx] * r[idx]) / W1[idx].sum()
for j, g in enumerate(GAM):
    d = M[j] - M[0]; se = S.boot_se(d)
    print(f"gamma {g:4.2f}: plain {np.nanmean(M[j]):.4f} ({np.nanmean(d):+.4f}, SE {se[0]:.4f}) | weighted {wm(d, a):+.4f} (SE {se[1]:.4f}) | f<0.3 {np.nanmean(d[SUB]):+.4f} (SE {se[2]:.4f})")
held = np.full(len(S.SEL), np.nan); ch = []
for f in range(3):
    tr, te = np.flatnonzero((fold != f) & keepw), np.flatnonzero((fold == f) & keepw)
    j = int(np.argmax([wm(M[j], tr) for j in range(len(GAM))])); ch.append(GAM[j]); held[te] = M[j, te]
d = held - M[0]; se = S.boot_se(d)
print(f"nested {ch}: plain {np.nanmean(d):+.4f} (SE {se[0]:.4f}) | weighted {wm(np.nan_to_num(d), a):+.4f} (SE {se[1]:.4f}) | f<0.3 {np.nanmean(d[SUB]):+.4f} (SE {se[2]:.4f})")
