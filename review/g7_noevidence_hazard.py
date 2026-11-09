"""Gem check G7 (P1 applied to the ongoing near region): queues at links with no same-month evidence (no queue in the
same slot on up to 3 earlier days of the month, observations 43.5% hidden as in the test months) are new or incident
queues, which dissolve faster. Does the model over-predict their persistence? Train ongoing windows (selector-like),
s31-like blend OOF. persist = queued near cells at T+30 / queued cells at T; model = expected queued near cells / at T.
If the model over-predicts for no-evidence windows, shrinking their near probabilities is a transportable correction."""
import sys, numpy as np, pandas as pd
sys.path.insert(0, r"C:/Users/user/tfb/src")
import t2_eval_ongoing_shift as O
from t2_eval_oof import T, load_oof
from cachelib import load

SPEC = "l127_full_af:2+q14_attn1_glob+q16_attnglob_s1+q14_attn1_wp05:2+q13_attn1:2+q12_af_r13_nopos+q17_drop1"
oof = load_oof(SPEC); PC = {}; rng = np.random.default_rng(0)
names = list(np.load("E:/tfb_nn/nn_data_afdem/names.npy", allow_pickle=True)); ci = names.index("last_queue")
X = np.load("E:/tfb_nn/nn_data_afdem/X.npy", mmap_mode="r"); OFF = np.load("E:/tfb_nn/nn_data_afdem/off.npy")


def bestset(p):
    pe = p.ravel(); o = np.argsort(-pe); cs = np.cumsum(pe[o]); kk = np.arange(1, pe.size + 1)
    kb = int(np.argmax(cs / (kk + pe.sum() - cs))) + 1; s = np.zeros(pe.size, bool); s[o[:kb]] = True
    return s.reshape(p.shape)


def iou(s, lab, el):
    u = ((s | lab) & el).sum()
    return 1.0 if u == 0 else (s & lab & el).sum() / u


rows = []
for k in O.SEL:
    if k not in oof:
        continue
    p = str(T.META["panel"][k]); t = int(T.META["t"][k]); day = str(T.META["date"][k])[:10]
    if p not in PC:
        P = load(p, "train"); q = (P.V <= P.vcut[None, :]) & np.isfinite(P.V); PC[p] = q & (rng.random(q.shape) > 0.435)
    q = PC[p]
    lq = np.asarray(X[OFF[k]:OFF[k + 1], ci], dtype=float) == 1
    near = np.zeros(len(lq), bool)
    for j in np.flatnonzero(lq):
        near[max(0, j - 3):j + 4] = True
    nprior = 0; ev = False
    for back in (1, 2, 3):
        dd = t - back * 288
        if dd - 18 >= 0 and str(np.datetime64(day) - np.timedelta64(back, "D"))[:7] == day[:7]:
            nprior += 1; ev |= bool(q[dd - 18:dd + 12][:, lq].any()) if lq.any() else False
    if nprior == 0 or not lq.any():
        continue
    lab, el = T.LAB[k]; lab = lab.astype(bool); el = el.astype(bool); pk = oof[k]
    n0 = lq.sum(); actual = (lab[-1] & near).sum() / n0; model = pk[-1][near].sum() / n0
    r = [iou(bestset(pk), lab, el)]
    for a in (0.8, 0.6, 0.45, 0.3):                                     # shrink the near-region probabilities of no-evidence queues
        pp = pk.copy(); pp[:, near] *= a; r.append(iou(bestset(pp), lab, el))
    rows.append((ev, actual, model, *r, pd.Timestamp(day).dayofweek >= 5))
R = pd.DataFrame(rows, columns=["evidence", "actual", "model", "iou", "iou_a08", "iou_a06", "iou_a045", "iou_a03", "weekend"])
for ev, g in R.groupby("evidence"):
    print(f"evidence={ev!s:5s} n={len(g):4d} persist actual {g.actual.mean():.3f}  model {g.model.mean():.3f}  (model-actual {g.model.mean()-g.actual.mean():+.3f}) | "
          f"IoU {g.iou.mean():.4f}  shrink 0.8 {g.iou_a08.mean()-g.iou.mean():+.4f}  0.6 {g.iou_a06.mean()-g.iou.mean():+.4f}  0.45 {g.iou_a045.mean()-g.iou.mean():+.4f}  0.3 {g.iou_a03.mean()-g.iou.mean():+.4f}")
print(f"share of windows without same-month evidence: {1 - R.evidence.mean():.3f}")
