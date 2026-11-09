"""Gem check G5 (knowledge-map gap: T2 ongoing near x same-month earlier days): does the way queues at the same links
evolved in the same slot on up to 3 earlier days of the month (persist ratio = queued cells at T+30 / at T in the near
region) explain what the model misses today? Train ongoing windows (selector-like), s31-like blend OOF.
residual = actual persist ratio - model-predicted persist ratio; we correlate it with the prior-day ratio."""
import sys, numpy as np, pandas as pd
sys.path.insert(0, r"C:/Users/user/tfb/src")
import t2_eval_ongoing_shift as O
from t2_eval_oof import T, load_oof
from cachelib import load

SPEC = "l127_full_af:2+q14_attn1_glob+q16_attnglob_s1+q14_attn1_wp05:2+q13_attn1:2+q12_af_r13_nopos+q17_drop1"
oof = load_oof(SPEC); PC = {}
names = list(np.load("E:/tfb_nn/nn_data_afdem/names.npy", allow_pickle=True)); ci = names.index("last_queue")
X = np.load("E:/tfb_nn/nn_data_afdem/X.npy", mmap_mode="r"); OFF = np.load("E:/tfb_nn/nn_data_afdem/off.npy")
rows = []
for k in O.SEL:
    if k not in oof:
        continue
    p = str(T.META["panel"][k]); t = int(T.META["t"][k]); day = str(T.META["date"][k])[:10]
    if p not in PC:
        P = load(p, "train"); PC[p] = (P.V <= P.vcut[None, :]) & np.isfinite(P.V)
    q = PC[p]
    lq = np.asarray(X[OFF[k]:OFF[k + 1], ci], dtype=float) == 1
    near = np.zeros(len(lq), bool)
    for j in np.flatnonzero(lq):
        near[max(0, j - 3):j + 4] = True
    lab, el = T.LAB[k]; lab = lab.astype(bool)
    n0 = max(lq.sum(), 1)
    actual = (lab[-1] & near).sum() / n0
    model = oof[k][-1][near].sum() / n0                      # expected queued near cells at T+30 / queued at T
    pr = []
    for back in (1, 2, 3):
        dd = t - back * 288
        if dd - 1 >= 0 and str(np.datetime64(day) - np.timedelta64(back, "D"))[:7] == day[:7]:
            a0 = q[dd - 1][near].sum()
            if a0 > 0:
                pr.append(q[dd + 6][near].sum() / a0)
    rows.append((actual, model, np.mean(pr) if pr else np.nan, len(pr), pd.Timestamp(day).dayofweek >= 5))
R = pd.DataFrame(rows, columns=["actual", "model", "prior", "nprior", "weekend"]).dropna()
R["resid"] = R.actual - R.model
print(f"windows with prior-day queues at the same links: {len(R)} / {len(rows)}")
print(f"corr(actual, model) {np.corrcoef(R.actual, R.model)[0,1]:.3f}   corr(actual, prior) {np.corrcoef(R.actual, R.prior)[0,1]:.3f}   "
      f"corr(residual, prior) {np.corrcoef(R.resid, R.prior)[0,1]:.3f}")
b = np.polyfit(R.prior, R.resid, 1); print(f"residual ~ prior slope {b[0]:+.3f}; mean |residual| {R.resid.abs().mean():.3f}")
for lab_, m in (("weekday", ~R.weekend), ("weekend", R.weekend)):
    r = R[m]; print(f"  {lab_}: n={len(r)} corr(residual, prior) {np.corrcoef(r.resid, r.prior)[0,1]:.3f}")
