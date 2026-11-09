"""G20: does the validation composition explain the onset LB gap (local 0.9075 vs LB 0.827)? No-memory blend OOF +
the s45 onset evidence rule (43.5% hidden, prior days within the month), per window; grouped by weekend/weekday and
day-of-month (1 / 2-4 / later); reweighted to the validation and private onset compositions (window_index dates)."""
import sys, numpy as np, pandas as pd
sys.path.insert(0, r"C:/Users/user/tfb/src")
import t2_eval_surprise as S
from t2_eval_oof import T, load_oof
from cachelib import load, T2_PANELS
from common import REL

SPEC = sys.argv[1] if len(sys.argv) > 1 else "q25_nomem_s3+q25_nomem_s4+q31_nomem_wp05_s0:2+q31_nomem_attn1_s0:2+q31_nomem_r13_s0+q31_nomem_drop1_s0"
oof = load_oof(SPEC); PC = {}


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


rows = []
for k in S.SEL:
    if k not in oof:
        continue
    p = str(T.META["panel"][k]); t = int(T.META["t"][k]); day = pd.Timestamp(str(T.META["date"][k])[:10]); q = qobs(p)
    lab, el = T.LAB[k]; lab = lab.astype(bool); el = el.astype(bool); b = oof[k]
    seen = np.zeros(q.shape[1], bool); nd = 0
    for bk in (1, 2, 3):
        dd = t - bk * 288
        if dd - 6 >= 0 and (day - pd.Timedelta(days=bk)).month == day.month:
            seen |= q[dd - 6:dd + 12].any(axis=0); nd += 1
    ev = seen.copy()
    for o in (1, 2):
        ev[o:] |= seen[:-o]; ev[:-o] |= seen[o:]
    rows.append(dict(panel=p, we=day.dayofweek >= 5, dom=day.day, raw=iou_last(b[-1], b, lab, el),
                     rule=iou_last(b[-1] * np.where(ev, 1.0, 0.5), b, lab, el) if nd else iou_last(b[-1], b, lab, el)))
R = pd.DataFrame(rows); R["dgrp"] = np.where(R.dom == 1, "day1", np.where(R.dom <= 4, "day2-4", "later"))
print(R.groupby(["we", "dgrp"])[["raw", "rule"]].agg(["mean", "count"]).round(4))
# test compositions
C = []
for sp in ("validation", "private"):
    for pn in T2_PANELS:
        w = pd.read_csv(REL / "task2" / pn / sp / "window_index.csv"); w = w[w.condition == "queue_onset"]
        d = pd.to_datetime(w.date)
        for x in d:
            C.append((sp, x.dayofweek >= 5, "day1" if x.day == 1 else ("day2-4" if x.day <= 4 else "later")))
C = pd.DataFrame(C, columns=["split", "we", "dgrp"])
cell = R.groupby(["we", "dgrp"]).rule.mean()
for sp in ("validation", "private"):
    c = C[C.split == sp].groupby(["we", "dgrp"]).size(); c = c / c.sum()
    est = sum(c[i] * cell.get(i, np.nan) for i in c.index)
    print(sp, "composition", dict(c.round(3)), "-> local estimate (rule)", round(est, 4))
print("local all", R.rule.mean().round(4))
