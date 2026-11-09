"""G15: same-day evidence for onset windows (20% of the test onset windows are on day 1 and get no prior-day evidence).
The onset selector leaves rows before T-12 free, so earlier queues of the same day (05:00 .. T-13) are legal evidence.
Rules fixed a priori (P8, no search): same-day links (+-2) queued in that span; links without it x B (B = 0.5 as the
onset rule), only when the day has any such queue.
  S1 'as day 1': only same-day evidence (what a day-1 window would see), on every window -> value as a substitute
  S2 'union'   : prior-day slot evidence (s45 rule) OR same-day evidence, vs the s45 rule alone
No-memory blend OOF, observation layer 43.5% hidden, SE by panel x date clusters, per month fold."""
import sys, numpy as np, pandas as pd
sys.path.insert(0, r"C:/Users/user/tfb/src")
import t2_eval_surprise as S
from t2_eval_oof import T, load_oof
from cachelib import load

SPEC = "q25_nomem_s3+q25_nomem_s4+q31_nomem_wp05_s0:2+q31_nomem_attn1_s0:2+q31_nomem_r13_s0+q31_nomem_drop1_s0"
oof = load_oof(SPEC); rng = np.random.default_rng(0); PC = {}


def qobs(p):
    if p not in PC:
        P = load(p, "train"); q = (P.V <= P.vcut[None, :]) & np.isfinite(P.V)
        PC[p] = q & (np.random.default_rng(0).random(q.shape) > 0.435)
    return PC[p]


def dil(seen, d=2):
    ev = seen.copy()
    for o in range(1, d + 1):
        ev[o:] |= seen[:-o]; ev[:-o] |= seen[o:]
    return ev


def iou_last(p_last, base, lab, el):
    p = base.copy(); p[-1] = p_last
    pe = p.ravel(); o = np.argsort(-pe); cs = np.cumsum(pe[o]); kk = np.arange(1, pe.size + 1)
    kb = int(np.argmax(cs / (kk + pe.sum() - cs))) + 1; s = np.zeros(pe.size, bool); s[o[:kb]] = True; s = s.reshape(p.shape)
    u = ((s | lab) & el).sum()
    return 1.0 if u == 0 else (s & lab & el).sum() / u


R = {"none": [], "S1": [], "s45": [], "S2": []}; cl = []; has = []
for k in S.SEL:
    if k not in oof:
        continue
    p = str(T.META["panel"][k]); t = int(T.META["t"][k]); day = pd.Timestamp(str(T.META["date"][k])[:10]); q = qobs(p)
    lab, el = T.LAB[k]; lab = lab.astype(bool); el = el.astype(bool); b = oof[k]
    d0 = t - (t % 288) + 60
    same = q[d0:t - 12].any(axis=0) if t - 12 > d0 else np.zeros(q.shape[1], bool)
    prior = np.zeros(q.shape[1], bool); nd = 0
    for bk in (1, 2, 3):
        dd = t - bk * 288
        if dd - 6 >= 0 and (day - pd.Timedelta(days=bk)).month == day.month:
            prior |= q[dd - 6:dd + 12].any(axis=0); nd += 1
    R["none"].append(iou_last(b[-1], b, lab, el))
    R["S1"].append(iou_last(b[-1] * np.where(dil(same), 1.0, 0.5), b, lab, el) if same.any() else R["none"][-1])
    r45 = iou_last(b[-1] * np.where(dil(prior), 1.0, 0.5), b, lab, el) if nd else R["none"][-1]
    R["s45"].append(r45)
    un = prior | same
    R["S2"].append(iou_last(b[-1] * np.where(dil(un), 1.0, 0.5), b, lab, el) if (nd or same.any()) else R["none"][-1])
    cl.append(f"{p}|{day.date()}"); has.append(same.any())
R = {k: np.array(v) for k, v in R.items()}; cl = np.array(cl); has = np.array(has)


def boot(d, reps=400):
    u = np.unique(cl); g = {c: np.flatnonzero(cl == c) for c in u}
    return np.std([d[np.concatenate([g[c] for c in rng.choice(u, len(u))])].mean() for _ in range(reps)])


print(f"onset windows {len(cl)}, with same-day evidence (05:00..T-13) {has.mean():.2f}")
for a, b_ in (("S1", "none"), ("S2", "s45"), ("s45", "none")):
    d = R[a] - R[b_]; print(f"{a} - {b_}: {d.mean():+.4f} (SE {boot(d):.4f}) | on windows with same-day evidence {d[has].mean():+.4f}")
