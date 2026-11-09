"""G24: room for evidence-driven insertion on onset windows. With prior-day evidence (s45 rule, 43.5% hidden):
P(link queued at T+30 | evidence) and | no evidence; recall of the label links by evidence; for label links WITH
evidence, how often the model gives p < 0.3 (missed by the expected-IoU set) - the cases an insertion rule would fix.
Split by the surprise weight (rare site for the slot in other months, f<0.3)."""
import sys, numpy as np, pandas as pd
sys.path.insert(0, r"C:/Users/user/tfb/src")
import t2_eval_surprise as S
from t2_eval_oof import T, load_oof
from cachelib import load
oof = load_oof("q25_nomem_s3+q25_nomem_s4+q31_nomem_wp05_s0:2+q31_nomem_attn1_s0:2+q31_nomem_r13_s0+q31_nomem_drop1_s0"); PC = {}
def qobs(p):
    if p not in PC:
        P = load(p, "train"); q = (P.V <= P.vcut[None, :]) & np.isfinite(P.V)
        PC[p] = q & (np.random.default_rng(0).random(q.shape) > 0.435)
    return PC[p]
R = []
for i, k in enumerate(S.SEL):
    p = str(T.META["panel"][k]); t = int(T.META["t"][k]); day = pd.Timestamp(str(T.META["date"][k])[:10]); q = qobs(p)
    lab, el = T.LAB[k]; y = lab[-1].astype(bool) & el[-1].astype(bool); b = oof[k][-1]
    seen = np.zeros(q.shape[1], bool); nd = 0
    for bk in (1, 2, 3):
        dd = t - bk * 288
        if dd - 6 >= 0 and (day - pd.Timedelta(days=bk)).month == day.month:
            seen |= q[dd - 6:dd + 12].any(axis=0); nd += 1
    if not nd:
        continue
    ev = seen.copy()
    for o in (1, 2):
        ev[o:] |= seen[:-o]; ev[:-o] |= seen[o:]
    R.append(dict(sub=bool(S.SUB[i]), n_ev=ev.sum(), p_y_ev=y[ev].mean() if ev.any() else np.nan, p_y_noev=y[~ev].mean(),
                  recall=y[ev].sum() / max(y.sum(), 1), miss_ev=int((y & ev & (b < 0.3)).sum()), y_n=int(y.sum()),
                  win_has_missed_ev=bool((y & ev & (b < 0.3)).any()), p_lab_ev=b[y & ev].mean() if (y & ev).any() else np.nan))
R = pd.DataFrame(R)
print(R.groupby("sub")[["n_ev", "p_y_ev", "p_y_noev", "recall", "p_lab_ev", "win_has_missed_ev"]].mean().round(3))
print("windows with evidence:", len(R), " missed evidenced label links total:", R.miss_ev.sum(), "of", R.y_n.sum())
