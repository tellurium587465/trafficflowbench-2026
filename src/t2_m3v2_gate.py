"""M3 v2 scoring (10/22 evening; see m3v2.py): the M3 gate's post-processing (current evidence rule, far gate D=3 +
contiguous growth) with the selector-simulated blanks instead of trap 4, and every window scored as the expectation over
k (official distribution truncated at the window's day, capped at 3 like the production rule). Windows, probabilities and
labels are those of t2_m3_gate (imported). Also re-runs the calibration pairs (LB signs) under v2.
  score_onset(p_, m, rule)  / score_ongoing(p_, m, gate)  -> expected IoU over k
usage: python t2_m3v2_gate.py   (calibration pairs under v2; prints both target mixes)"""
import sys
if __name__ == "__main__":
    sys.argv = [sys.argv[0], "0,1,2,3"]
import numpy as np, pandas as pd
import t2_m3_gate as G
import m3v2
from t2_model import best_set


def _hidden_today(m, k):
    return m3v2.sim_blank_rows(m["panel"], m["t"] // 288, min(3, k), m["t"])


def onset_pred(p_, m, rule, k):
    pb = p_.copy()
    if rule != "none":
        s, nd = m3v2.seen(m["panel"], m["t"], k, -6, 12)
        if nd and not (rule == "skip" and not s.any()):
            pb[-1] = pb[-1] * np.where(G.dil(s, 2), 1.0, 0.25)
    return best_set(pb.ravel(), np.ones(pb.size, bool)).reshape(pb.shape)


def far_gate(sel, m, k):
    P, _, q = m3v2._panel(m["panel"]); t = m["t"]; d0 = t - (t % 288) + 60; hid = _hidden_today(m, k)
    rr = [r for r in range(d0, t) if r not in hid]
    ev = q[rr].any(axis=0) if rr else np.zeros(P.n, bool)
    s2, _ = m3v2.seen(m["panel"], t, k, -18, 12)
    ev = G.dil(ev | s2, 2)
    i = G.POS[m["window_id"]]; lq = np.asarray(G.X[G.OFF[i]:G.OFF[i + 1], G.CI], dtype=float) == 1
    near = np.zeros(P.n, bool)
    for j in np.flatnonzero(lq):
        near[max(0, j - 3):j + 4] = True
    C = np.zeros_like(sel)
    for r_ in range(sel.shape[0]):
        j = 0
        while j < sel.shape[1]:
            if sel[r_, j]:
                e = j
                while e + 1 < sel.shape[1] and sel[r_, e + 1]:
                    e += 1
                if near[j:e + 1].any():
                    C[r_, j:e + 1] = True
                j = e + 1
            else:
                j += 1
    return sel & ~(sel & ~near[None, :] & ~ev[None, :] & ~C)


def ongoing_pred(p_, m, gate, k):
    s, nd = m3v2.seen(m["panel"], m["t"], k, -18, 12)
    pb = p_ * np.where(G.dil(s, 1), 1.0, 0.85)[None, :] if nd else p_.copy()
    sel = best_set(pb.ravel(), np.ones(pb.size, bool)).reshape(pb.shape)
    return far_gate(sel, m, k) if gate else sel


def expect(m, fn):
    return sum(pr * fn(k) for k, pr in m3v2.kdist(m["day"], cap=3).items())


def score_onset(p_, m, rule="cur"):
    return expect(m, lambda k: G.iou(onset_pred(p_, m, rule, k), m))


def score_ongoing(p_, m, gate=True):
    return expect(m, lambda k: G.iou(ongoing_pred(p_, m, gate, k), m))


def score(p_, m):
    return score_onset(p_, m) if m["cond"] == "queue_onset" else score_ongoing(p_, m)


def frame(cols):
    """cols: {name: {window_id: score}} -> DataFrame with cond, panel, day, hour + the columns"""
    rows = []
    for w in G.WID:
        m = G.META[w]; r = dict(w=w, cond=m["cond"], panel=m["panel"], day=m["day"], hour=(m["t"] % 288) // 12)
        for n, c in cols.items():
            if w in c:
                r[n] = c[w]
        rows.append(r)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    import time
    t0 = time.time(); cols = {k: {} for k in ("mem", "nomem", "ev_none", "ev_cur", "skip", "gate_off", "gate_on", "sim", "neg")}
    for w in G.WID:
        m = G.META[w]
        if m["cond"] == "queue_onset":
            cols["mem"][w] = score_onset(G.MEM[w], m); cols["nomem"][w] = cols["ev_cur"][w] = score_onset(G.NOMEM[w], m)
            cols["ev_none"][w] = score_onset(G.NOMEM[w], m, "none"); cols["skip"][w] = score_onset(G.NOMEM[w], m, "skip"); cols["neg"][w] = score_onset(G.NEG[w], m)
        else:
            cols["mem"][w] = score_ongoing(G.MEM[w], m); cols["nomem"][w] = cols["gate_on"][w] = score_ongoing(G.NOMEM[w], m)
            cols["gate_off"][w] = score_ongoing(G.NOMEM[w], m, False); cols["sim"][w] = score_ongoing(G.SIM[w], m); cols["neg"][w] = score_ongoing(G.NEG[w], m)
    D = frame(cols); D.to_csv(G.W + f"m3v2_gate{G.VER}" + ("_L1" if G.LABEL == "L1" else "") + ".csv", index=False); print(f"scored {len(D)} windows in {time.time() - t0:.0f}s")
    pairs = [("1 memory -> no memory (LB +)", D, "mem", "nomem"), ("2 evidence rule (LB +)", D[D.cond == "queue_onset"], "ev_none", "ev_cur"),
             ("3 far gate (LB +)", D[D.cond == "queue_ongoing"], "gate_off", "gate_on"), ("4 sim ongoing (LB +)", D[D.cond == "queue_ongoing"], "nomem", "sim"),
             ("5 B1 skip (LB -)", D[D.cond == "queue_onset"], "ev_cur", "skip"), ("6 negative control (LB -)", D, "nomem", "neg")]
    for name, g, a, b in pairs:
        for tgt in ("validation", "private"):
            print(m3v2.fmt(name, m3v2.summary(g, a, b, tgt)))
