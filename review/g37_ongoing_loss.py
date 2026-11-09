"""G37: ongoing loss map (local, in-distribution) and simple physics baselines. Selector-like ongoing windows.
  nn       : OOF of the no-memory NN blend, expected-IoU set
  persist  : the queue at the last fully observed row (T-1) repeated over the 6 steps (T row not used)
  tailx    : persistence, but each queue run's upstream tail extended by the observed tail speed over the last 30 min
Per window IoU; split by whether the queue grows / shrinks / is stable (from the labels, diagnostic only).
usage: python g37_ongoing_loss.py"""
import sys, numpy as np, pandas as pd, warnings
warnings.simplefilter("ignore")
sys.path.insert(0, r"C:/Users/user/tfb/src")
from t2_eval_oof import T, load_oof
from cachelib import load
SPEC = "q25_nomem_s3+q25_nomem_s4+q31_nomem_wp05_s0:2+q31_nomem_attn1_s0:2+q31_nomem_r13_s0+q31_nomem_drop1_s0"
oof = load_oof(SPEC); PC = {}
SEL = [k for k in range(T.NWIN) if T.META["is_sel"][k] and T.META["cond"][k] == "queue_ongoing" and k in oof]


def bestset(p):
    pe = p.ravel(); o = np.argsort(-pe); cs = np.cumsum(pe[o]); kk = np.arange(1, pe.size + 1)
    kb = int(np.argmax(cs / (kk + pe.sum() - cs))) + 1; s = np.zeros(pe.size, bool); s[o[:kb]] = True; return s.reshape(p.shape)


def iou(s, y, el):
    u = ((s | y) & el).sum(); return 1.0 if u == 0 else (s & y & el).sum() / u


def tail_runs(q):
    """list of (start, end) of contiguous True runs"""
    x = np.flatnonzero(np.diff(np.concatenate([[0], q.astype(int), [0]])))
    return list(zip(x[::2], x[1::2]))


rows = []
for k in SEL:
    p = str(T.META["panel"][k]); t = int(T.META["t"][k])
    if p not in PC:
        P = load(p, "train"); PC[p] = (P, (P.V <= P.vcut[None, :]) & np.isfinite(P.V))
    P, Q = PC[p]; lab, el = T.LAB[k]; y = lab.astype(bool); e = el.astype(bool)
    now = Q[t - 1]; pers = np.repeat(now[None, :], 6, axis=0)
    # tail speed: upstream end of each run at t-1 vs t-7 (links per 30 min, upstream = lower index)
    tx = pers.copy()
    for a, bnd in tail_runs(now):
        prev = Q[t - 7, max(0, a - 6):bnd]
        a0 = max(0, a - 6) + int(np.flatnonzero(prev)[0]) if prev.any() else a
        rate = (a0 - a) / 6.0                              # links per row (positive = growing upstream)
        for s in range(6):
            na = int(round(a - rate * (s + 1)))
            na = min(max(na, 0), bnd - 1)
            tx[s, :] = tx[s, :]
            if na < a:
                tx[s, na:a] = True
            elif na > a:
                tx[s, a:na] = False
    gr = (y[-1] & e[-1]).sum() - (now & e[-1]).sum()
    rows.append(dict(panel=p, nn=iou(bestset(oof[k]), y, e), persist=iou(pers, y, e), tailx=iou(tx, y, e),
                     trend="grow" if gr > 1 else ("shrink" if gr < -1 else "stable")))
R = pd.DataFrame(rows)
print(f"ongoing windows {len(R)}: NN {R.nn.mean():.4f} | persistence {R.persist.mean():.4f} | tail extrapolation {R.tailx.mean():.4f}")
print(R.groupby("trend")[["nn", "persist", "tailx"]].agg(["mean", "size"]).round(3).to_string())
print(f"NN IoU<0.5 share {np.mean(R.nn < 0.5):.3f}, zero {np.mean(R.nn == 0):.3f}")
