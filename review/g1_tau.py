"""Gem check G1 (M3 B3): does the arrival-time score tau_i = (C_i - q_i) / max(eps, dq_i/dt) rank the onset link?
Train onset windows of the selector-like set (t2_eval_surprise.SEL). Inputs use rows <= T-1 (the T row is excluded as in
the models). Truth = links queued at T+30. Per window: AUC of the score over links and top-1 hit within +-2 links.
Compared with the nomem NN OOF probability at T+30, on all windows and on rare-site windows (f < 0.3), and with a
simple rank average of the two (complementarity)."""
import sys, numpy as np
sys.path.insert(0, r"C:/Users/user/tfb/src")
import t2_eval_surprise as S
from t2_eval_oof import T, load_oof
from cachelib import load

NN = load_oof("q23_nomem_s3+q23_nomem_s4")
PC = {}


def auc(score, pos):
    if pos.all() or not pos.any():
        return np.nan
    r = np.argsort(np.argsort(score)) + 1.0
    n1 = pos.sum(); n0 = (~pos).sum()
    return (r[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def hit(score, pos, tol=2):
    i = int(np.argmax(score)); t = np.flatnonzero(pos)
    return float(np.min(np.abs(t - i)) <= tol) if len(t) else np.nan


rows = []
for j, k in enumerate(S.SEL):
    p = str(T.META["panel"][k]); t = int(T.META["t"][k])
    if p not in PC:
        P = load(p, "train"); PC[p] = (np.where(P.E, P.Q, np.nan).astype(np.float32), P.cap.astype(np.float32))
    Q, cap = PC[p]
    h = Q[max(0, t - 12):t]                                     # 60 min before T, T row excluded
    if len(h) < 6:
        continue
    x = np.arange(len(h), dtype=np.float32); ok = np.isfinite(h)
    qn = np.array([np.nanmean(h[-3:, i]) if ok[-3:, i].any() else np.nan for i in range(h.shape[1])])
    sl = np.full(h.shape[1], np.nan)
    for i in range(h.shape[1]):
        m = ok[:, i]
        if m.sum() >= 4:
            sl[i] = np.polyfit(x[m], h[m, i], 1)[0] * 12.0           # veh/h per hour
    tau = (cap - qn) / np.maximum(sl, 50.0)                        # hours to reach capacity (rising flows only)
    tau = np.where(np.isfinite(tau), tau, 99.0)
    util = np.where(np.isfinite(qn), qn / cap, 0.0)
    lab = T.LAB[k][0][-1].astype(bool)
    if k not in NN or not lab.any():
        continue
    nn = NN[k][-1]
    sc_tau, sc_util = -tau, util
    comb = (np.argsort(np.argsort(nn)) + np.argsort(np.argsort(sc_tau))) / 2.0
    rows.append((S.F[j] < 0.3, auc(nn, lab), auc(sc_tau, lab), auc(sc_util, lab), auc(comb, lab),
                 hit(nn, lab), hit(sc_tau, lab), hit(sc_util, lab), hit(comb, lab)))

R = np.array(rows, dtype=float)
names = ["AUC nn", "AUC tau", "AUC util", "AUC nn+tau", "hit nn", "hit tau", "hit util", "hit nn+tau"]
for lab_, m in (("all onset", np.ones(len(R), bool)), ("rare sites f<0.3", R[:, 0] == 1)):
    print(f"{lab_:18s} n={m.sum():4d} " + "  ".join(f"{n} {np.nanmean(R[m, i + 1]):.3f}" for i, n in enumerate(names)))
