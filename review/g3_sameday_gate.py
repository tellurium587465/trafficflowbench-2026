"""Gem check G3 (knowledge-map gap: T2 x same-day pre-T evidence): keep a far ongoing prediction only where a queue
was already observed earlier the same day (from 05:00 up to T-5, causal) within +-2 links; drop the other far
predictions. Train ongoing windows (selector-like), s31-like blend OOF. Compares: no far rule / far rule (D=3) / gated.
Split by weekday-weekend because the LB (validation) is weekend-heavy and the private month is weekday-heavy."""
import sys, numpy as np, pandas as pd
sys.path.insert(0, r"C:/Users/user/tfb/src")
import os
os.environ.setdefault("TFB_T2_FAR", "3")
import t2_eval_ongoing_shift as O
from t2_eval_oof import T, load_oof, far_mask
from cachelib import load

MODE = sys.argv[1] if len(sys.argv) > 1 else "day"
SPEC = "l127_full_af:2+q14_attn1_glob+q16_attnglob_s1+q14_attn1_wp05:2+q13_attn1:2+q12_af_r13_nopos+q17_drop1"
oof = load_oof(SPEC); PC = {}


def bestset(p):
    pe = p.ravel(); o = np.argsort(-pe); cs = np.cumsum(pe[o]); kk = np.arange(1, pe.size + 1)
    kb = int(np.argmax(cs / (kk + pe.sum() - cs))) + 1; s = np.zeros(pe.size, bool); s[o[:kb]] = True
    return s.reshape(p.shape)


def iou(s, lab, el):
    u = ((s | lab) & el).sum()
    return 1.0 if u == 0 else (s & lab & el).sum() / u


rows = []
for j, k in enumerate(O.SEL):
    if k not in oof:
        continue
    p = str(T.META["panel"][k]); t = int(T.META["t"][k])
    if p not in PC:
        P = load(p, "train"); PC[p] = (P.V <= P.vcut[None, :]) & np.isfinite(P.V)
    q = PC[p]; d0 = t - (t % 288) + 60                            # 05:00 of the same day
    seen = q[d0:t].any(axis=0) if t > d0 else np.zeros(q.shape[1], bool)
    for back in (1, 2, 3):                                     # same slot on up to 3 earlier days of the same month (test windows are on days 1-4)
        dd = t - back * 288
        if dd - 18 >= 0 and str(T.META["date"][k])[:7] == str(np.datetime64(str(T.META["date"][k])[:10]) - np.timedelta64(back, "D"))[:7]:
            seen = seen | q[dd - 18:dd + 12].any(axis=0) if MODE == "month" else seen
    ev = seen.copy()
    for o in (1, 2):
        ev[o:] |= seen[:-o]; ev[:-o] |= seen[o:]
    lab, el = T.LAB[k]; lab = lab.astype(bool); el = el.astype(bool)
    s = bestset(oof[k]); far = far_mask(k)
    s_far = s & ~far[None, :]
    s_gate = s & ~(far & ~ev)[None, :]
    wk = pd.Timestamp(str(T.META["date"][k])).dayofweek >= 5
    rows.append((wk, iou(s, lab, el), iou(s_far, lab, el), iou(s_gate, lab, el),
                 (s & far[None, :]).sum(), (s & (far & ev)[None, :]).sum()))
R = pd.DataFrame(rows, columns=["weekend", "none", "far", "gated", "far_pos", "far_kept"])
for lab_, m in (("all", R.index == R.index), ("weekday", ~R.weekend), ("weekend", R.weekend)):
    r = R[m]
    print(f"{lab_:8s} n={len(r):4d} IoU no rule {r.none.mean():.4f} | far rule {r.far.mean():.4f} ({r.far.mean()-r.none.mean():+.4f}) | "
          f"same-day gate {r.gated.mean():.4f} ({r.gated.mean()-r.none.mean():+.4f}) | far positives kept by the gate {r.far_kept.sum()/max(r.far_pos.sum(),1):.2f}")
