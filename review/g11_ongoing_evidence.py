"""G11 (knowledge-map gap: T2 ongoing NEAR region x same-month evidence). For ongoing windows, probabilities of links
with same-month evidence (queued in the same slot T-90..T+60 min on up to 3 earlier days, +-dil links; observation
layer 43.5% hidden) are multiplied by A, the others by B, before the expected-IoU set; then the far gate as in s40.
Model: the s46-like no-memory blend OOF. Nested CV over the 3 month folds (choose (A, B, dil) on two folds, score the
third) with cluster-bootstrap SE, on the official-like ongoing windows (exact persistence IoU <= 0.9) and on all."""
import sys, os, itertools, numpy as np, pandas as pd
sys.path.insert(0, r"C:/Users/user/tfb/src")
os.environ.setdefault("TFB_T2_FAR", "3")
import t2_eval_ongoing_shift as O
from t2_eval_oof import T, load_oof, far_mask
from cachelib import load

SPEC = sys.argv[1] if len(sys.argv) > 1 else "q25_nomem_s3+q25_nomem_s4+q31_nomem_wp05_s0:2+q31_nomem_attn1_s0:2+q31_nomem_r13_s0+q31_nomem_drop1_s0"
oof = load_oof(SPEC); PC = {}; rng = np.random.default_rng(0)
per = np.load("E:/tfb_nn/persist_iou_afdem.npy")
FOLD = {**{m: 0 for m in ("2030-06", "2030-07", "2030-08")}, **{m: 1 for m in ("2030-09", "2030-10", "2030-11")},
        **{m: 2 for m in ("2030-12", "2031-01", "2031-02")}}


def qobs(p):
    if p not in PC:
        P = load(p, "train"); q = (P.V <= P.vcut[None, :]) & np.isfinite(P.V)
        PC[p] = q & (np.random.default_rng(0).random(q.shape) > 0.435)
    return PC[p]


def bestset(p):
    pe = p.ravel(); o = np.argsort(-pe); cs = np.cumsum(pe[o]); kk = np.arange(1, pe.size + 1)
    kb = int(np.argmax(cs / (kk + pe.sum() - cs))) + 1; s = np.zeros(pe.size, bool); s[o[:kb]] = True
    return s.reshape(p.shape)


def iou(s, lab, el):
    u = ((s | lab) & el).sum()
    return 1.0 if u == 0 else (s & lab & el).sum() / u


W = []
for k in O.SEL:
    if k not in oof:
        continue
    p = str(T.META["panel"][k]); t = int(T.META["t"][k]); day = pd.Timestamp(str(T.META["date"][k])[:10]); q = qobs(p)
    seen = np.zeros(q.shape[1], bool); nd = 0
    for b in (1, 2, 3):
        dd = t - b * 288
        if dd - 18 >= 0 and (day - pd.Timedelta(days=b)).month == day.month:
            seen |= q[dd - 18:dd + 12].any(axis=0); nd += 1
    d0 = t - (t % 288) + 60; gseen = seen | (q[d0:t].any(axis=0) if t > d0 else False)    # gate evidence as in s40
    gev = gseen.copy()
    for o in (1, 2):
        gev[o:] |= gseen[:-o]; gev[:-o] |= gseen[o:]
    lab, el = T.LAB[k]
    W.append(dict(k=k, base=oof[k], seen=seen, nd=nd, gate=far_mask(k) & ~gev, lab=lab.astype(bool), el=el.astype(bool),
                  fold=FOLD[str(T.META["date"][k])[:7]], off=per[k] <= 0.9, cl=f"{p}|{day.date()}"))


def score(w, rule):
    pp = w["base"]
    if rule is not None and w["nd"]:
        A, B, dil = rule; ev = w["seen"].copy()
        for o in range(1, dil + 1):
            ev[o:] |= w["seen"][:-o]; ev[:-o] |= w["seen"][o:]
        pp = pp * np.where(ev, A, B)[None, :]
        pp = np.clip(pp, 0, 1)
    s = bestset(pp) & ~w["gate"][None, :]
    return iou(s, w["lab"], w["el"])


RULES = [None] + list(itertools.product((1.0, 1.2, 1.5), (1.0, 0.85, 0.7, 0.5), (0, 1, 2, 3)))
RULES = [r for r in RULES if r is None or not (r[0] == 1.0 and r[1] == 1.0)]
M = np.array([[score(w, r) for w in W] for r in RULES]); fold = np.array([w["fold"] for w in W]); off = np.array([w["off"] for w in W])
cl = np.array([w["cl"] for w in W])


def boot(d, idx, reps=400):
    u = np.unique(cl[idx]); g = {c: idx[cl[idx] == c] for c in u}
    return np.std([d[np.concatenate([g[c] for c in rng.choice(u, len(u))])].mean() for _ in range(reps)])


for lab_, sub in (("all ongoing", np.ones(len(W), bool)), ("official-like (persistence<=0.9)", off)):
    held = np.zeros(len(W)); chosen = []
    for f in range(3):
        tr = np.flatnonzero((fold != f) & sub); te = np.flatnonzero((fold == f) & sub)
        b = int(np.argmax(M[:, tr].mean(axis=1))); chosen.append(RULES[b]); held[te] = M[b, te]
    idx = np.flatnonzero(sub); d = held - M[0]
    print(f"[{lab_}] n={len(idx)} gate only {M[0, idx].mean():.4f} | nested tuned {held[idx].mean():.4f} ({d[idx].mean():+.4f}, SE {boot(d, idx):.4f}) | chosen {chosen}")
    top = np.argsort(-M[:, idx].mean(axis=1))[:4]; print("   in-sample top:", [(RULES[i], round(M[i, idx].mean() - M[0, idx].mean(), 4)) for i in top])
