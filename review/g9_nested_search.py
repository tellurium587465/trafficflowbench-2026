"""Nested-CV random search of the onset same-month evidence rule (10/10, user request: tune hyperparameters with
k-fold before trusting them). Model: no-memory NN OOF (q25_nomem s3+s4), on which the local yardstick can see the
value of evidence (CAUSAL_MAP P7). Test-like: observation layer 43.5% hidden, at most 3 earlier days.
Space: slot start lo in {-18,-12,-6,0} steps, slot end hi in {6,12,18}, dilation {0..3}, earlier days {1,2,3},
day type {all, same}, weight B for links without evidence in {0.3,0.4,0.5,0.6,0.7,0.85}  (1728 combos; 300 random +
the current s45 rule). Outer loop: the 3 month folds; the rule is chosen on two folds and scored on the third, against
the current rule and against no rule. SE: bootstrap over panel x date clusters of the held-out per-window differences.
NOTE: the ongoing far-gate parameters are NOT tuned here: on train months far predictions are right (memory and
evidence agree), so local tuning would always favour no gate (CAUSAL_MAP P4)."""
import sys, os, itertools, numpy as np, pandas as pd
sys.path.insert(0, r"C:/Users/user/tfb/src")
import t2_eval_surprise as S
from t2_eval_oof import T, load_oof
from cachelib import load

SPEC = sys.argv[1] if len(sys.argv) > 1 else "q25_nomem_s3+q25_nomem_s4"
oof = load_oof(SPEC); rng = np.random.default_rng(0); PC = {}
FOLD = {**{m: 0 for m in ("2030-06", "2030-07", "2030-08")}, **{m: 1 for m in ("2030-09", "2030-10", "2030-11")},
        **{m: 2 for m in ("2030-12", "2031-01", "2031-02")}}
LO, HI, DIL, BACK, DT, BW = (-18, -12, -6, 0), (6, 12, 18), (0, 1, 2, 3), (1, 2, 3), (0, 1), (0.3, 0.4, 0.5, 0.6, 0.7, 0.85)
DEFAULT = (-6, 12, 2, 3, 0, 0.5)


def qobs(p):
    if p not in PC:
        P = load(p, "train"); q = (P.V <= P.vcut[None, :]) & np.isfinite(P.V)
        PC[p] = q & (np.random.default_rng(0).random(q.shape) > 0.435)
    return PC[p]


def iou_last(p_last, base, lab, el):
    p = base.copy(); p[-1] = p_last
    pe = p.ravel(); o = np.argsort(-pe); cs = np.cumsum(pe[o]); kk = np.arange(1, pe.size + 1)
    kb = int(np.argmax(cs / (kk + pe.sum() - cs))) + 1; s = np.zeros(pe.size, bool); s[o[:kb]] = True; s = s.reshape(p.shape)
    u = ((s | lab) & el).sum()
    return 1.0 if u == 0 else (s & lab & el).sum() / u


# per window: queued-any per (earlier day b, 30-min sub-bin from -18 to +18 steps), and the window metadata
W = []
for k in S.SEL:
    if k not in oof:
        continue
    p = str(T.META["panel"][k]); t = int(T.META["t"][k]); day = pd.Timestamp(str(T.META["date"][k])[:10]); q = qobs(p)
    bins = {}
    for b in (1, 2, 3):
        dd = t - b * 288; pday = day - pd.Timedelta(days=b)
        if dd - 18 >= 0 and pday.month == day.month:
            bins[b] = (np.stack([q[dd + a:dd + a + 6].any(axis=0) for a in range(-18, 18, 6)]), (pday.dayofweek >= 5) == (day.dayofweek >= 5))
    lab, el = T.LAB[k]
    W.append(dict(k=k, base=oof[k], lab=lab.astype(bool), el=el.astype(bool), bins=bins, fold=FOLD[str(T.META["date"][k])[:7]],
                  cl=f"{p}|{str(T.META['date'][k])[:10]}"))


def score_rule(w, rule):
    if rule is None:
        return iou_last(w["base"][-1], w["base"], w["lab"], w["el"])
    lo, hi, dil, back, dt, B = rule
    seen = np.zeros(w["base"].shape[1], bool); nd = 0
    for b, (bb, same) in w["bins"].items():
        if b > back or (dt and not same):
            continue
        a0, a1 = (lo + 18) // 6, (hi + 18) // 6
        seen |= bb[a0:a1].any(axis=0); nd += 1
    if not nd:
        return iou_last(w["base"][-1], w["base"], w["lab"], w["el"])
    ev = seen.copy()
    for o in range(1, dil + 1):
        ev[o:] |= seen[:-o]; ev[:-o] |= seen[o:]
    return iou_last(w["base"][-1] * np.where(ev, 1.0, B), w["base"], w["lab"], w["el"])


combos = list(itertools.product(LO, HI, DIL, BACK, DT, BW))
pick = [combos[i] for i in rng.choice(len(combos), 300, replace=False)]
if DEFAULT not in pick:
    pick.append(DEFAULT)
none = np.array([score_rule(w, None) for w in W])
M = np.array([[score_rule(w, r) for w in W] for r in pick])          # (rules, windows)
fold = np.array([w["fold"] for w in W]); cl = np.array([w["cl"] for w in W]); di = pick.index(DEFAULT)


def boot_se(d, idx, reps=400):
    u = np.unique(cl[idx]); g = {c: idx[cl[idx] == c] for c in u}; out = []
    for _ in range(reps):
        ii = np.concatenate([g[c] for c in rng.choice(u, len(u))]); out.append(d[ii].mean())
    return np.std(out)


held_tuned, held_def, held_none, chosen = np.zeros(len(W)), np.zeros(len(W)), np.zeros(len(W)), []
for f in range(3):
    tr, te = np.flatnonzero(fold != f), np.flatnonzero(fold == f)
    best = int(np.argmax(M[:, tr].mean(axis=1))); chosen.append(pick[best])
    held_tuned[te] = M[best, te]; held_def[te] = M[di, te]; held_none[te] = none[te]
allidx = np.arange(len(W))
d1 = held_tuned - held_def; d2 = held_def - held_none
print(f"onset windows {len(W)}, rules searched {len(pick)}")
print(f"chosen per outer fold (lo, hi, dil, back, daytype, B): {chosen}")
print(f"held-out: no rule {held_none.mean():.4f} | current s45 rule {held_def.mean():.4f} | tuned {held_tuned.mean():.4f}")
print(f"current - none {d2.mean():+.4f} (SE {boot_se(d2, allidx):.4f}) | tuned - current {d1.mean():+.4f} (SE {boot_se(d1, allidx):.4f})")
top = np.argsort(-M.mean(axis=1))[:5]
print("top 5 on all windows (in-sample, for reference):", [(pick[i], round(M[i].mean() - none.mean(), 4)) for i in top])
