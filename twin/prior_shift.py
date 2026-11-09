"""Prior-shift correction for onset (literature A/B, 10/12): instead of 'links without same-month evidence x B',
correct by how this month's evidence departs from the training-month frequency at that link and slot:
  f_i  = training-world share of days with a queue at link i (+-2 links) in the slot T-30..T+60 min (same day type)
  k_i  = earlier days of this month (<= 3) with a queue there in the masked layer, n = number of such days
  fhat = (k_i + a f_i) / (n + a)          (shrinkage toward the training frequency)
  r_i  = clip(fhat / f_i, 0.02, 50) ** b  -> p'(T+30) = p x r_i
A link that queued often in training but not this month is pushed down hard; a link rare in training that queued
this month is pushed up; uninformative links stay. Model: A-only NN on the test worlds T (the analogue of the real
onset model). Nested over world halves, (a, b) grid; vs the current rule (B = 0.25). Frequencies from world A.
usage: python prior_shift.py"""
import sys, pickle, itertools, numpy as np, pandas as pd
sys.path.insert(0, r"C:/Users/user/tfb/twin"); sys.path.insert(0, r"C:/Users/user/tfb/src")
from adjuster import probs, bestset, world, dil
from cachelib import load

LT = pickle.load(open("E:/tfb_nn/twin_labels.pkl", "rb")); PR = probs("tw2_nomem_s0+tw2_nomem_s1")
A_, B_ = (0.5, 1.0, 2.0, 4.0), (0.25, 0.5, 0.75, 1.0)
GRID = list(itertools.product(A_, B_))
FREQ = {}


def train_freq(panel):
    """(2 day types, 288 slots, n links): share of world-A days with a queue (+-2 links) within slot-6..slot+12"""
    if panel not in FREQ:
        P = load(panel, "twinA"); D = P.T // 288
        q = ((np.nan_to_num(P.V, nan=1e9) <= P.vcut[None, :]) & P.E)[:D * 288].reshape(D, 288, P.n)
        we = np.array([pd.Timestamp(str(P.day[d])).dayofweek >= 5 for d in range(D)])
        F = np.zeros((2, 288, P.n))
        for s in range(288):
            a, b = max(0, s - 6), min(288, s + 12)
            hit = q[:, a:b].any(axis=1)                                           # (D, n)
            hit = hit | np.roll(hit, 1, 1) | np.roll(hit, -1, 1) | np.roll(hit, 2, 1) | np.roll(hit, -2, 1)
            for w in (0, 1):
                F[w, s] = hit[we == w].mean(axis=0) if (we == w).any() else 0
        FREQ[panel] = F
    return FREQ[panel]


rows = []
for wid, rec in LT.items():
    y, el, panel, wd, t, cond, date = rec
    if cond != "queue_onset":
        continue
    P, obs, full = world(panel, wd); Y = y.T.astype(bool); EL = el.T.astype(bool)
    days = []
    for b in (1, 2, 3):
        dd = t - b * 288
        if dd - 6 >= 0:
            days.append(dil(obs[dd - 6:dd + 12].any(axis=0), 2))
    n = len(days); k = np.sum(days, axis=0) if n else np.zeros(P.n)
    we = int(pd.Timestamp(date).dayofweek >= 5); f = np.clip(train_freq(panel)[we, t % 288], 0.02, 1.0)
    p0 = PR[wid].copy(); p0[:-1] = 0
    r = dict(cl=panel + wd, half=int(wd[1:]) // 4, day=int(date[8:10]))
    def score(p):
        s = bestset(p); u = ((s | Y) & EL).sum(); return 1.0 if u == 0 else (s & Y & EL).sum() / u
    pc = p0.copy()
    if n:
        pc[-1] = pc[-1] * np.where(dil(np.any(days, axis=0), 0), 1.0, 0.25)       # current rule (days already dilated)
    r["cur"] = score(pc)
    for i, (a, b) in enumerate(GRID):
        p = p0.copy()
        if n:
            fhat = (k + a * f) / (n + a); p[-1] = np.clip(p[-1] * np.clip(fhat / f, 0.02, 50) ** b, 0, 1)
        r[i] = score(p)
    rows.append(r)
R = pd.DataFrame(rows); rng = np.random.default_rng(0)
top = np.argsort([-R[i].mean() for i in range(len(GRID))])[:4]
print("in-sample top (a, b) vs current:", [(GRID[i], round(R[i].mean() - R["cur"].mean(), 4)) for i in top])
held = np.zeros(len(R)); ch = []
for h in (0, 1):
    tr, te = (R.half != h).to_numpy(), (R.half == h).to_numpy()
    i = max(range(len(GRID)), key=lambda i: R[i].to_numpy()[tr].mean()); ch.append(GRID[i]); held[te] = R[i].to_numpy()[te]
d = held - R["cur"].to_numpy(); cl = R.cl.to_numpy(); u = np.unique(cl); idx = {c: np.flatnonzero(cl == c) for c in u}
se = np.std([d[np.concatenate([idx[c] for c in rng.choice(u, len(u))])].mean() for _ in range(500)])
m = ((R.day >= 2) & (R.day <= 4)).to_numpy()
print(f"nested: chosen {ch}, held-out - current {d.mean():+.4f} (SE {se:.4f}); days 2-4 {d[m].mean():+.4f} (n {m.sum()})")
