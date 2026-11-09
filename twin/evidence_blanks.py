"""Onset evidence rule under the release blanks (10/13). In the test months every scored window hides rows T+5..T+90 of
all links, and onset windows sit one per morning on days 1-5, so the production rule's evidence (earlier days, slots
T-30..T+60, +-2 links) mostly falls into the earlier day's blank: 43 of 80 validation/private onset windows have NO
visible evidence and the rule then scales every link by 0.25 (a uniform shrink, not evidence). The twin's earlier rule
checks did not reproduce the blanks (template blanks, not aligned with the twin's own windows).
Here every labelled twin window (onset and ongoing) blanks rows T+1..T+18 of its world, then onset rule variants:
  none      : no rule
  cur       : production (slots T-30..T+60 on <= 3 earlier days, +-2 links; x B where no evidence; applied if nd > 0)
  cur_skip  : as cur, but no scaling when no link has evidence
  after     : evidence = queue heads (queued link, downstream free) in T+60..T+180 on the earlier days, +-2 links
  union     : cur or after evidence; no scaling when neither shows anything
  shrink    : every link x B whenever nd > 0 (the uniform part alone)
B in (0.25, 0.5). A-trained NN (tw2_nomem s0+s1) on the T worlds (+ L worlds when their CPU probs exist); IoU at the
onset step, cluster bootstrap over panel x world; days 2-5 reported separately (where evidence can exist).
usage: python evidence_blanks.py"""
import sys, glob, pickle, numpy as np, pandas as pd, warnings
warnings.simplefilter("ignore")
sys.path.insert(0, r"C:/Users/user/tfb/twin"); sys.path.insert(0, r"C:/Users/user/tfb/src")
from adjuster import bestset, dil
from cachelib import load
W = "C:/Users/user/tfb/work/t2/"


def blend(paths):
    P = [pickle.load(open(p, "rb")) for p in paths]
    return {k: sum(p[k] for p in P) / len(P) for k in P[0]}


WC = {}


def world_blanked(panel, wd, LT_world):
    key = (panel, wd)
    if key not in WC:
        P = load(panel, f"twin{wd}")
        q = (P.V <= P.vcut[None, :]) & np.isfinite(P.V) & ~P.blank
        hide = np.zeros(P.T, bool)
        for t in LT_world:
            hide[t + 1:t + 19] = True
        q = q & ~hide[:, None]
        head = q & ~np.concatenate([q[:, 1:], np.zeros((P.T, 1), bool)], axis=1)
        WC[key] = (type("Pl", (), {"n": P.n})(), q, head)       # keep only what the rule needs (memory, 10/14)
        load.cache_clear()
    return WC[key]


def evaluate(LT, PR, tag):
    by_world = {}
    for wid, rec in LT.items():
        by_world.setdefault((rec[2], rec[3]), []).append(rec[4])
    rows = []
    for wid, rec in LT.items():
        y, el, panel, wd, t, cond, date = rec
        if cond != "queue_onset" or wid not in PR or (t + 6) % 288 >= 144:   # review 10/13: morning onsets only
            continue
        P, q, head = world_blanked(panel, wd, by_world[(panel, wd)])
        Y = y.T.astype(bool); EL = el.T.astype(bool)
        days = [t - b * 288 for b in (1, 2, 3) if t - b * 288 - 6 >= 0]
        nd = len(days)
        ev_cur = dil(np.any([q[dd - 6:dd + 12].any(axis=0) for dd in days], axis=0), 2) if nd else np.zeros(P.n, bool)
        ev_aft = dil(np.any([head[dd + 12:dd + 36].any(axis=0) for dd in days], axis=0), 2) if nd else np.zeros(P.n, bool)
        p0 = PR[wid].copy(); p0[:-1] = 0
        def iou(p):
            s = bestset(p); u = ((s | Y) & EL).sum(); return 1.0 if u == 0 else (s & Y & EL).sum() / u
        r = dict(cl=panel + wd, day=int(date[8:10]), nd=nd, has_cur=bool(ev_cur.any()), has_aft=bool(ev_aft.any()), none=iou(p0))
        for B in (0.25, 0.5):
            def ap(ev, skip):
                p = p0.copy()
                if nd and not (skip and not ev.any()):
                    p[-1] = p[-1] * np.where(ev, 1.0, B)
                return iou(p)
            r[f"cur{B}"] = ap(ev_cur, False); r[f"cur_skip{B}"] = ap(ev_cur, True)
            r[f"after{B}"] = ap(ev_aft, True); r[f"union{B}"] = ap(ev_cur | ev_aft, True)
            p = p0.copy()
            if nd:
                p[-1] = p[-1] * B
            r[f"shrink{B}"] = iou(p)
        rows.append(r)
    R = pd.DataFrame(rows); rng = np.random.default_rng(0)
    cols = [c for c in R.columns if c not in ("cl", "day", "nd", "has_cur", "has_aft")]
    m25 = (R.day >= 2) & (R.day <= 5)
    print(f"[{tag}] onset windows {len(R)}; with earlier days {int((R.nd > 0).sum())}: visible evidence cur {R[R.nd > 0].has_cur.mean():.2f}, after-blank {R[R.nd > 0].has_aft.mean():.2f}")
    u = R.cl.unique(); idx = {x: np.flatnonzero(R.cl.to_numpy() == x) for x in u}
    for c in cols:
        d = (R[c] - R["cur0.25"]).to_numpy()
        se = np.std([d[np.concatenate([idx[x] for x in rng.choice(u, len(u))])].mean() for _ in range(300)])
        print(f"   {c:14s} {R[c].mean():.4f}  vs production {d.mean():+.4f} (SE {se:.4f}) | days 2-5 {d[m25.to_numpy()].mean():+.4f}")
    return R


LT = pickle.load(open("E:/tfb_nn/twin_labels.pkl", "rb"))
evaluate(LT, blend([W + f"probs_tw2_nomem_s{s}.pkl" for s in (0, 1)]), "T worlds, A-trained")
