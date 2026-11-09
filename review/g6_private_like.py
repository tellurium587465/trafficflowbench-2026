"""Private-like yardstick (10/10): the private windows are on days 1-4 of the month and 82.5% weekday (validation:
60-65% weekday). Restrict the train ongoing / onset selector-like windows to day-of-month <= 4 and compare, on weekday
and weekend separately:
  far rule  : none / blanket (D=3) / same-month evidence gate (as t2_far_gate.py), evidence computed from observations
              with 43.5% of cells hidden at random (the validation/private observation layer is 43.5% missing)
  models    : s31-like blend, attention base (q23_base s3+s4), no-memory NN (q23_nomem s3+s4)
"""
import sys, os, numpy as np, pandas as pd
sys.path.insert(0, r"C:/Users/user/tfb/src")
os.environ.setdefault("TFB_T2_FAR", "3")
import t2_eval_ongoing_shift as O, t2_eval_surprise as S
from t2_eval_oof import T, load_oof, far_mask
from cachelib import load

MODELS = {"s31-like": "l127_full_af:2+q14_attn1_glob+q16_attnglob_s1+q14_attn1_wp05:2+q13_attn1:2+q12_af_r13_nopos+q17_drop1",
          "base NN": "q23_base_s3+q23_base_s4", "nomem NN": "q23_nomem_s3+q23_nomem_s4"}
if len(sys.argv) > 1:                                          # name=spec ... overrides the default models
    MODELS = dict(a.split("=", 1) for a in sys.argv[1:])
OOF = {m: load_oof(s) for m, s in MODELS.items()}
rng = np.random.default_rng(0); PC = {}


def qmask(p):
    if p not in PC:
        P = load(p, "train"); q = (P.V <= P.vcut[None, :]) & np.isfinite(P.V)
        PC[p] = q & (rng.random(q.shape) > 0.435)              # observation layer as sparse as in the test months
    return PC[p]


def bestset(p):
    pe = p.ravel(); o = np.argsort(-pe); cs = np.cumsum(pe[o]); kk = np.arange(1, pe.size + 1)
    kb = int(np.argmax(cs / (kk + pe.sum() - cs))) + 1; s = np.zeros(pe.size, bool); s[o[:kb]] = True
    return s.reshape(p.shape)


def iou(s, lab, el):
    u = ((s | lab) & el).sum()
    return 1.0 if u == 0 else (s & lab & el).sum() / u


def evidence(k):
    p = str(T.META["panel"][k]); t = int(T.META["t"][k]); q = qmask(p); day = str(T.META["date"][k])[:10]
    d0 = t - (t % 288) + 60; seen = q[d0:t].any(axis=0) if t > d0 else np.zeros(q.shape[1], bool)
    for back in (1, 2, 3):
        dd = t - back * 288
        if dd - 18 >= 0 and str(np.datetime64(day) - np.timedelta64(back, "D"))[:7] == day[:7]:
            seen = seen | q[dd - 18:dd + 12].any(axis=0)
    ev = seen.copy()
    for o in (1, 2):
        ev[o:] |= seen[:-o]; ev[:-o] |= seen[o:]
    return ev


rows = []
for k in O.SEL:
    day = pd.Timestamp(str(T.META["date"][k]))
    if day.day > 4:
        continue
    lab, el = T.LAB[k]; lab = lab.astype(bool); el = el.astype(bool); far = far_mask(k); ev = evidence(k)
    for m, o in OOF.items():
        if k not in o:
            continue
        s = bestset(o[k])
        rows.append(("ongoing", m, day.dayofweek >= 5, iou(s, lab, el), iou(s & ~far[None, :], lab, el),
                     iou(s & ~(far & ~ev)[None, :], lab, el)))
for k in S.SEL:
    day = pd.Timestamp(str(T.META["date"][k]))
    if day.day > 4:
        continue
    lab, el = T.LAB[k]; lab = lab.astype(bool); el = el.astype(bool)
    for m, o in OOF.items():
        if k in o:
            v = iou(bestset(o[k]), lab, el); rows.append(("onset", m, day.dayofweek >= 5, v, v, v))
R = pd.DataFrame(rows, columns=["cond", "model", "weekend", "none", "blanket", "gate"])
for cond in ("ongoing", "onset"):
    for wk, lab_ in ((False, "weekday"), (True, "weekend")):
        r = R[(R.cond == cond) & (R.weekend == wk)]
        if not len(r):
            continue
        g = r.groupby("model")[["none", "blanket", "gate"]].mean()
        n = r.groupby("model").size().iloc[0]
        print(f"[{cond} {lab_} day<=4, n={n}]")
        for m, x in g.iterrows():
            print(f"   {m:9s} none {x.none:.4f}  blanket {x.blanket:.4f} ({x.blanket-x.none:+.4f})  gate {x.gate:.4f} ({x.gate-x.none:+.4f})" if cond == "ongoing" else f"   {m:9s} {x.none:.4f}")
