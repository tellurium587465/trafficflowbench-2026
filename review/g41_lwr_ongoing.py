"""G41: a site-agnostic LWR forecast of ongoing queues vs the NN, by how rare the site is in the other train months
(real data, selector-like ongoing windows; the rare-site windows stand in for the map shift). For every queue run at
T-1 (queued links, gaps of one link closed): head h fixed at the bottleneck; tail a moves at the shock speed
  w = (q_queue - q_in) / (k_queue - k_up)   (km/h, negative = growing upstream)
q_in, k_up: median flow / density on links a-3..a-1 over T-3..T-1 (free flow upstream); q_queue, k_queue: inside the run.
Tail after s steps = a - w * 5s/60 h converted to links by the link lengths; a run whose tail passes its head clears.
Also a blend: NN probabilities x (1 + lam * LWR set) renormalised by the expected-IoU set.
usage: python g41_lwr_ongoing.py"""
import sys, numpy as np, pandas as pd, warnings
warnings.simplefilter("ignore")
sys.argv = ["x"]
exec(open(r"C:/Users/user/tfb/review/g37_ongoing_loss.py", encoding="utf-8").read().split("rows = []")[0])
FC = {}
g30 = open(r"C:/Users/user/tfb/review/g30_onset_priorshift.py", encoding="utf-8").read()
exec(g30[g30.index("def month_counts"):g30.index("def iou_last")])


def runs(x):
    y = x.copy()
    y[1:-1] |= x[:-2] & x[2:]                                         # close one-link gaps
    idx = np.flatnonzero(np.diff(np.concatenate([[0], y.astype(int), [0]])))
    return list(zip(idx[::2], idx[1::2] - 1))                         # (tail, head) inclusive


def lwr(P, t):
    Q, V = P.Q, P.V; q = (V <= P.vcut[None, :]) & np.isfinite(V); now = q[t - 1] | (q[t - 2] & q[t])  if False else q[t - 1]
    L = np.nan_to_num(P.length, nan=0.1); out = np.zeros((6, P.n), bool)
    for a, h in runs(now):
        with np.errstate(all="ignore"):
            up = slice(max(0, a - 3), a)
            q_in = np.nanmedian(Q[t - 3:t, up]); v_up = np.nanmedian(V[t - 3:t, up])
            q_qu = np.nanmedian(Q[t - 3:t, a:h + 1]); v_qu = np.nanmedian(V[t - 3:t, a:h + 1])
        if not np.isfinite([q_in, v_up, q_qu, v_qu]).all() or a == 0:
            out[:, a:h + 1] = True; continue
        k_up = q_in / max(v_up, 1); k_qu = q_qu / max(v_qu, 1)
        w = (q_qu - q_in) / max(k_qu - k_up, 1e-3)                    # km/h; < 0: tail moves upstream
        for s in range(6):
            dist = -w * 5 * (s + 1) / 60                               # km the tail moves upstream (negative: downstream)
            if dist >= 0:
                tail = a; acc = 0.0
                while tail > 0 and acc + L[tail - 1] <= dist:
                    acc += L[tail - 1]; tail -= 1
                out[s, tail:h + 1] = True
            else:
                tail = a; acc = 0.0
                while tail <= h and acc + L[tail] <= -dist:
                    acc += L[tail]; tail += 1
                if tail <= h:
                    out[s, tail:h + 1] = True
    return out


rows = []
for k in SEL:
    p = str(T.META["panel"][k]); t = int(T.META["t"][k]); day = pd.Timestamp(str(T.META["date"][k])[:10])
    if p not in PC:
        P = load(p, "train"); PC[p] = (P, (P.V <= P.vcut[None, :]) & np.isfinite(P.V))
    P, Qm = PC[p]; lab, el = T.LAB[k]; y = lab.astype(bool); e = el.astype(bool)
    yy = (y & e).any(axis=0)
    if not yy.any():
        continue
    pers = np.repeat(Qm[t - 1][None, :], 6, axis=0)
    if iou(pers, y, e) > 0.9:
        continue
    f = freq_excl(p, str(T.META["date"][k])[:7], day.dayofweek >= 5)[t % 288]
    Lw = lwr(P, t); pn = oof[k]
    r = dict(f=float(np.mean(f[yy])), nn=iou(bestset(pn), y, e), lwr=iou(Lw, y, e), pers=iou(pers, y, e))
    for lam in (0.25, 0.5, 1.0):
        r[f"blend{lam}"] = iou(bestset(np.clip(pn * (1 + lam * Lw), 0, 1)), y, e)
    rows.append(r)
R = pd.DataFrame(rows); R["fb"] = pd.cut(R.f, [-0.01, 0.1, 0.3, 0.6, 1.01])
cols = ["nn", "lwr", "pers", "blend0.25", "blend0.5", "blend1.0"]
print(f"official-type ongoing windows {len(R)}"); print(R[cols].mean().round(4).to_string())
print(R.groupby("fb", observed=True)[cols].mean().round(4).to_string()); print(R.groupby("fb", observed=True).size().to_string())
