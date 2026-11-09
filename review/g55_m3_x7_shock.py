"""G55: X7 reachability cone + shock-wave extrapolation for ongoing windows (STATE.md 10/22 pre-registration;
literature/D_top_approaches.md section 4-3 and section 7 #5). M3 ongoing windows, NOMEM blend (q24f, seeds 0-3).
For each run of links visible as queued at T (last_queue): head = downstream end (E / N panels: higher link index,
W / S: lower; checked on the topology), tail per history row T-6..T-1 = the most upstream observed congested link reached
from the head through congested-or-unobserved links (read rows < T only). Tail length L (km from the head) fitted by
least squares over the rows, speed s clipped to |s| <= w = 25 km/h (2.08 km per 5-min step); fewer than 3 rows -> s = 0.
  V1 cone : at T+h, cells of the upstream continuation farther than L(T) + w h from the head (up to the next run) x0.5
  V2 shock: the extrapolated run (head .. L(T) + s h upstream) as a 0/1 prediction, averaged 50:50 with the NN
then the current post-processing (OGB x0.85, expected-IoU set, far gate). Constants fixed in advance (not tuned).
usage: python g55_m3_x7_shock.py"""
import sys
sys.path.insert(0, r"C:/Users/user/tfb/src")
sys.argv = [sys.argv[0], "0,1,2,3"]
import numpy as np, pandas as pd
import t2_m3_gate as G
from m3_stats import summary, fmt

WSTEP = 25.0 * 5 / 60                                          # km per 5-min step
WOG = [w for w in G.WID if G.META[w]["cond"] == "queue_ongoing"]


def runs(v):
    out, j = [], 0
    while j < len(v):
        if v[j]:
            e = j
            while e + 1 < len(v) and v[e + 1]:
                e += 1
            out.append((j, e)); j = e + 1
        else:
            j += 1
    return out


def parts(w):
    m = G.META[w]; P, q = G.panel(m["panel"]); t = m["t"]; n = P.n; obs = np.isfinite(P.V)
    up = -1                                  # 10/22 fix: low -> high index in every panel (audit A12)
    xc = np.cumsum(P.length) - P.length / 2
    i = G.POS[w]; lq = np.asarray(G.X[G.OFF[i]:G.OFF[i + 1], G.CI], dtype=float) == 1
    S = np.zeros((6, n)); cone = np.ones((6, n)); stats = []
    for a, b in runs(lq):
        head = b if up == -1 else a
        pts = []
        for r in range(t - 6, t):
            j, last = head, None
            while 0 <= j < n and not (obs[r, j] and not q[r, j]):
                if obs[r, j] and q[r, j]:
                    last = j
                j += up
            if last is not None:
                pts.append((r - t, abs(xc[last] - xc[head])))
        if len(pts) >= 3:
            xs, ls = np.array(pts, float).T; s, L0 = np.polyfit(xs, ls, 1); s = float(np.clip(s, -WSTEP, WSTEP))
            L0 = float(np.mean(ls - s * xs))                    # intercept at T with the clipped slope
        else:
            s = 0.0; tl = a if up == -1 else b; L0 = abs(xc[tl] - xc[head])
        stats.append((len(pts), s))
        j_end = head                                            # upstream continuation: up to the next visible run
        while 0 <= j_end + up < n and not lq[j_end + up] or (0 <= j_end + up < n and a <= j_end + up <= b):
            j_end += up
        js = np.arange(head, j_end + up, up) if j_end != head else np.array([head])
        d = np.abs(xc[js] - xc[head])
        for h in range(1, 7):
            S[h - 1, js[d <= L0 + s * h]] = 1.0
            cone[h - 1, js[d > L0 + WSTEP * h]] = 0.5
    return S, cone, stats


rows, st = [], []
for w in WOG:
    m = G.META[w]; p = G.NOMEM[w]; S, cone, s_ = parts(w); st += s_
    rows.append(dict(cond=m["cond"], panel=m["panel"], day=m["day"], cur=G.iou(G.ongoing_pred(p, m, True), m),
                     cone=G.iou(G.ongoing_pred(p * cone, m, True), m), shock=G.iou(G.ongoing_pred(0.5 * p + 0.5 * S, m, True), m)))
D = pd.DataFrame(rows); D.to_csv(G.W + "g55_m3_x7_shock.csv", index=False)
st = np.array(st)
print(f"visible runs {len(st)} | tail tracked (>= 3 rows) {np.mean(st[:, 0] >= 3):.2f} | speed s (km/step): median {np.median(st[:, 1]):+.3f}, "
      f"growing {np.mean(st[:, 1] > 0.05):.2f}, shrinking {np.mean(st[:, 1] < -0.05):.2f}, clipped {np.mean(np.abs(st[:, 1]) >= WSTEP - 1e-9):.2f}")
print(fmt("V1 cone x0.5 vs current (ongoing)", summary(D, "cur", "cone")))
print(fmt("V2 shock 50:50 vs current (ongoing)", summary(D, "cur", "shock")))
