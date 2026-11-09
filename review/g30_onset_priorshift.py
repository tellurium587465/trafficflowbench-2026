"""G30: onset prior-shift correction (twin/prior_shift.py) on the real train months, before it is used as the second
final (FINAL_SELECTION.md, hedge H2). Same windows, OOF and observation thinning (43.5% hidden) as G23.
  f_i  = share of OTHER train months' days (same day type) with a queue at link i (+-2) in slots T-6..T+11 (unthinned,
         as world A in the twin)
  k_i  = earlier days of this month (<= 3) with a queue there in the thinned layer, n = number of such days
  p'(T+30) = p x clip((k + a f) / (n + a) / f, 0.02, 50) ** b
vs the production rule (links without evidence x 0.25). In the train months the map does not move, so the question is
only 'no significant harm' (P11 condition 2), on plain / surprise-weighted / f<0.3 means; nested (a, b) over the 3 folds."""
import sys, itertools, numpy as np, pandas as pd
sys.path.insert(0, r"C:/Users/user/tfb/src")
import t2_eval_surprise as S
from t2_eval_oof import T, load_oof
from cachelib import load

SPEC = sys.argv[1] if len(sys.argv) > 1 else "q25_nomem_s3+q25_nomem_s4+q31_nomem_wp05_s0:2+q31_nomem_attn1_s0:2+q31_nomem_r13_s0+q31_nomem_drop1_s0"
oof = load_oof(SPEC); PC = {}; FC = {}
GRID = list(itertools.product((0.5, 1.0, 2.0, 4.0), (0.25, 0.5, 0.75, 1.0)))
FOLD = {**{m: 0 for m in ("2030-06", "2030-07", "2030-08")}, **{m: 1 for m in ("2030-09", "2030-10", "2030-11")},
        **{m: 2 for m in ("2030-12", "2031-01", "2031-02")}}


def qobs(p):
    if p not in PC:
        P = load(p, "train"); q = (P.V <= P.vcut[None, :]) & np.isfinite(P.V)
        PC[p] = q & (np.random.default_rng(0).random(q.shape) > 0.435)
    return PC[p]


def month_counts(p):
    """per (month, day type): number of days, and (288, n) count of days with a queue (+-2 links) in slots s-6..s+11"""
    if p not in FC:
        P = load(p, "train"); D = P.T // 288
        q = ((P.V <= P.vcut[None, :]) & np.isfinite(P.V))[:D * 288].reshape(D, 288, P.n).astype(np.int32)
        cs = np.concatenate([np.zeros((D, 1, P.n), np.int32), np.cumsum(q, axis=1)], axis=1)   # (D, 289, n)
        s = np.arange(288); lo, hi = np.maximum(0, s - 6), np.minimum(288, s + 12)
        hit = (cs[:, hi] - cs[:, lo]) > 0                                                       # (D, 288, n)
        e = hit.copy()
        for o in (1, 2):
            e[:, :, o:] |= hit[:, :, :-o]; e[:, :, :-o] |= hit[:, :, o:]
        mon = np.array([d[:7] for d in P.day[:D]]); we = np.array([pd.Timestamp(str(d)).dayofweek >= 5 for d in P.day[:D]])
        FC[p] = {(m, w): (int(((mon == m) & (we == w)).sum()), e[(mon == m) & (we == w)].sum(axis=0))
                 for m in np.unique(mon) for w in (False, True)}
    return FC[p]


def freq_excl(p, month, w):
    C = month_counts(p); n = sum(v[0] for (m, ww), v in C.items() if ww == w and m != month)
    c = sum(v[1] for (m, ww), v in C.items() if ww == w and m != month)
    return c / max(n, 1)


def iou_last(pl, base, lab, el):
    p = base.copy(); p[-1] = pl
    pe = p.ravel(); o = np.argsort(-pe); cs = np.cumsum(pe[o]); kk = np.arange(1, pe.size + 1)
    kb = int(np.argmax(cs / (kk + pe.sum() - cs))) + 1; s = np.zeros(pe.size, bool); s[o[:kb]] = True; s = s.reshape(p.shape)
    u = ((s | lab) & el).sum()
    return 1.0 if u == 0 else (s & lab & el).sum() / u


def dil2(x):
    e = x.copy()
    for o in (1, 2):
        e[o:] |= x[:-o]; e[:-o] |= x[o:]
    return e


CUR = np.zeros(len(S.SEL)); M = np.full((len(GRID), len(S.SEL)), np.nan); fold = np.zeros(len(S.SEL), int)
nds = np.zeros(len(S.SEL), int); wkend = np.zeros(len(S.SEL), bool)
for i, k in enumerate(S.SEL):
    p = str(T.META["panel"][k]); t = int(T.META["t"][k]); day = pd.Timestamp(str(T.META["date"][k])[:10]); q = qobs(p)
    lab, el = T.LAB[k]; lab = lab.astype(bool); el = el.astype(bool); b = oof[k]; mon = str(T.META["date"][k])[:7]
    fold[i] = FOLD[mon]; wkend[i] = day.dayofweek >= 5
    days = []
    for bk in (1, 2, 3):
        dd = t - bk * 288
        if dd - 6 >= 0 and (day - pd.Timedelta(days=bk)).month == day.month:
            days.append(dil2(q[dd - 6:dd + 12].any(axis=0)))
    n = len(days); nds[i] = n
    ev = np.any(days, axis=0) if n else None; kc = np.sum(days, axis=0) if n else None
    CUR[i] = iou_last(b[-1] * (np.where(ev, 1.0, 0.25) if n else 1.0), b, lab, el)
    f = np.clip(freq_excl(p, mon, bool(wkend[i]))[t % 288], 0.02, 1.0)
    for j, (a, bb) in enumerate(GRID):
        pl = b[-1] if not n else np.clip(b[-1] * np.clip((kc + a * f) / (n + a) / f, 0.02, 50) ** bb, 0, 1)
        M[j, i] = iou_last(pl, b, lab, el)
W1, SUB = S.W1, S.SUB
wm = lambda r, idx: np.sum(W1[idx] * r[idx]) / W1[idx].sum()
a_ = np.arange(len(S.SEL))
print(f"windows {len(S.SEL)}, with earlier days {int((nds > 0).sum())}; current (B=0.25): plain {CUR.mean():.4f} / weighted {wm(CUR, a_):.4f} / f<0.3 {CUR[SUB].mean():.4f}")
for j in np.argsort([-wm(M[j], a_) for j in range(len(GRID))])[:6]:
    print(f"  (a, b) = {GRID[j]}: diff plain {M[j].mean() - CUR.mean():+.4f} / weighted {wm(M[j], a_) - wm(CUR, a_):+.4f} / f<0.3 {M[j][SUB].mean() - CUR[SUB].mean():+.4f}")
j05 = GRID.index((0.5, 1.0)); d = M[j05] - CUR; se = S.boot_se(d)
print(f"hedge setting (0.5, 1): plain {d.mean():+.4f} (SE {se[0]:.4f}) | weighted {wm(d, a_):+.4f} (SE {se[1]:.4f}) | f<0.3 {d[SUB].mean():+.4f} (SE {se[2]:.4f}) | weekday {d[~wkend].mean():+.4f}, weekend {d[wkend].mean():+.4f}")
held = np.zeros(len(S.SEL)); ch = []
for fo in range(3):
    tr, te = np.flatnonzero(fold != fo), np.flatnonzero(fold == fo)
    j = int(np.argmax([wm(M[j], tr) for j in range(len(GRID))])); ch.append(GRID[j]); held[te] = M[j, te]
d = held - CUR; se = S.boot_se(d)
print(f"nested (chosen by weighted) {ch}: plain {d.mean():+.4f} (SE {se[0]:.4f}) | weighted {wm(d, a_):+.4f} (SE {se[1]:.4f}) | f<0.3 {d[SUB].mean():+.4f} (SE {se[2]:.4f})")
