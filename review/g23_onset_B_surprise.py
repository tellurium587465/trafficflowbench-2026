"""G23: onset evidence strength B judged on the surprise yardstick (t2_eval_surprise: weight 1/max(f,0.05) by how rare
the true site is for the slot in other months, and the f<0.3 subset that reproduced the LB level). Fixed structure
(s45: T-30..T+60 min, +-2 links, up to 3 earlier days in the month, 43.5% hidden); B in a 1-D scan. Nested over the
3 month folds: B chosen by the weighted metric on two folds, scored on the third."""
import sys, numpy as np, pandas as pd
sys.path.insert(0, r"C:/Users/user/tfb/src")
import t2_eval_surprise as S
from t2_eval_oof import T, load_oof
from cachelib import load

SPEC = sys.argv[1] if len(sys.argv) > 1 else "q25_nomem_s3+q25_nomem_s4+q31_nomem_wp05_s0:2+q31_nomem_attn1_s0:2+q31_nomem_r13_s0+q31_nomem_drop1_s0"
oof = load_oof(SPEC); PC = {}
BS = (1.0, 0.7, 0.5, 0.35, 0.25, 0.15, 0.05)
FOLD = {**{m: 0 for m in ("2030-06", "2030-07", "2030-08")}, **{m: 1 for m in ("2030-09", "2030-10", "2030-11")},
        **{m: 2 for m in ("2030-12", "2031-01", "2031-02")}}


def qobs(p):
    if p not in PC:
        P = load(p, "train"); q = (P.V <= P.vcut[None, :]) & np.isfinite(P.V)
        PC[p] = q & (np.random.default_rng(0).random(q.shape) > 0.435)
    return PC[p]


def iou_last(pl, base, lab, el):
    p = base.copy(); p[-1] = pl
    pe = p.ravel(); o = np.argsort(-pe); cs = np.cumsum(pe[o]); kk = np.arange(1, pe.size + 1)
    kb = int(np.argmax(cs / (kk + pe.sum() - cs))) + 1; s = np.zeros(pe.size, bool); s[o[:kb]] = True; s = s.reshape(p.shape)
    u = ((s | lab) & el).sum()
    return 1.0 if u == 0 else (s & lab & el).sum() / u


M = np.full((len(BS), len(S.SEL)), np.nan); fold = np.zeros(len(S.SEL), int); hasev = np.zeros(len(S.SEL), bool)
for i, k in enumerate(S.SEL):
    p = str(T.META["panel"][k]); t = int(T.META["t"][k]); day = pd.Timestamp(str(T.META["date"][k])[:10]); q = qobs(p)
    lab, el = T.LAB[k]; lab = lab.astype(bool); el = el.astype(bool); b = oof[k]; fold[i] = FOLD[str(T.META["date"][k])[:7]]
    seen = np.zeros(q.shape[1], bool); nd = 0
    for bk in (1, 2, 3):
        dd = t - bk * 288
        if dd - 6 >= 0 and (day - pd.Timedelta(days=bk)).month == day.month:
            seen |= q[dd - 6:dd + 12].any(axis=0); nd += 1
    ev = seen.copy()
    for o in (1, 2):
        ev[o:] |= seen[:-o]; ev[:-o] |= seen[o:]
    hasev[i] = nd > 0
    for j, B in enumerate(BS):
        M[j, i] = iou_last(b[-1] * (np.where(ev, 1.0, B) if nd else 1.0), b, lab, el)
W1, SUB = S.W1, S.SUB
wm = lambda r, idx: np.sum(W1[idx] * r[idx]) / W1[idx].sum()
print("B scan (in-sample): plain / weighted / f<0.3")
for j, B in enumerate(BS):
    a = np.arange(len(S.SEL)); print(f"  B={B:4.2f}: {M[j].mean():.4f} / {wm(M[j], a):.4f} / {M[j][SUB].mean():.4f}")
held = np.zeros(len(S.SEL)); ch = []
for f in range(3):
    tr, te = np.flatnonzero(fold != f), np.flatnonzero(fold == f)
    j = int(np.argmax([wm(M[j], tr) for j in range(len(BS))])); ch.append(BS[j]); held[te] = M[j, te]
cur = M[BS.index(0.5)]; d = held - cur
se = S.boot_se(d)
print(f"nested (chosen by weighted) {ch}: held - current(B=0.5) plain {d.mean():+.4f} (SE {se[0]:.4f}) | weighted {wm(d, np.arange(len(d))):+.4f} (SE {se[1]:.4f}) | f<0.3 {d[SUB].mean():+.4f} (SE {se[2]:.4f})")
d0 = cur - M[0]; se0 = S.boot_se(d0)
print(f"current - none: plain {d0.mean():+.4f} (SE {se0[0]:.4f}) | weighted {wm(d0, np.arange(len(d0))):+.4f} (SE {se0[1]:.4f}) | f<0.3 {d0[SUB].mean():+.4f} (SE {se0[2]:.4f})")
