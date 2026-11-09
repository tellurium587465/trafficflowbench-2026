"""Reachability cone for ongoing windows (10/22, X7 V3; adopted on M3 under the statistical-only rejection rule):
same computation as review/g55_m3_x7_shock.py (cone) + review/g57_x7_cone_ev.py (evidence exemption), for production.
For each run of links visible as queued at T (last_queue): head = downstream end (the higher link index in every panel:
audit/a12_direction.py; until 10/22 evening the W / S panels were traced the wrong way), tail per row T-6..T-1 = the most upstream observed congested link reached from the head through congested-or-
unobserved links; tail length L fitted by least squares, speed clipped to 25 km/h (2.08 km per 5-min step; < 3 rows ->
s = 0). At T+h the upstream continuation (up to the next visible run) farther than L(T) + w h from the head gets x0.5,
except links with same-month evidence (a queue the same day 05:00 .. T-1 or in the slot T-90 .. T+60 min on up to 3
earlier days, +-2 links; as t2_far_gate.py). Reads rows < T only (cachelib.asof).
  apply : python t2_cone.py <probs_src> <probs_dst>   (work/t2/probs_<name>.pkl; ongoing windows x factor, onset as is)
       env TFB_T2_WINDOWS=<id>[,<id>...] / TFB_T2_TEST=<nn_test dir> (10/23, t2_window_entry.py): only those windows, their
       features (last_queue) from that directory
       env TFB_T2_LQT=1 (11/03 ledger #197): a link observed at the row T takes its queue state there as the last queue"""
import os, sys, pickle, numpy as np, pandas as pd
from cachelib import load, T2_PANELS, asof
from common import REL, NN, WORK

WSTEP = 25.0 * 5 / 60


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


def dil(s, r):
    e = s.copy()
    for o in range(1, r + 1):
        e[o:] |= s[:-o]; e[:-o] |= s[o:]
    return e


def cone(P, q, obs, t, lq, panel):
    """(6, n) cone factor (1 or 0.5) without the evidence exemption; q / obs: (T, n) congested / observed, rows < t used"""
    n = P.n; xc = np.cumsum(P.length) - P.length / 2
    up = -1                                       # 10/22 fix: traffic flows low -> high index in every panel (audit A12); the topology table is reversed for W/S
    out = np.ones((6, n))
    for a, b in runs(lq):
        head = b if up == -1 else a; pts = []
        for r in range(t - 6, t):
            j, last = head, None
            while 0 <= j < n and not (obs[r, j] and not q[r, j]):
                if obs[r, j] and q[r, j]:
                    last = j
                j += up
            if last is not None:
                pts.append((r - t, abs(xc[last] - xc[head])))
        if len(pts) >= 3:
            xs, ls = np.array(pts, float).T; s, _ = np.polyfit(xs, ls, 1); s = float(np.clip(s, -WSTEP, WSTEP)); L0 = float(np.mean(ls - s * xs))
        else:
            tl = a if up == -1 else b; L0 = abs(xc[tl] - xc[head])
        j_end = head
        while 0 <= j_end + up < n and not lq[j_end + up] or (0 <= j_end + up < n and a <= j_end + up <= b):
            j_end += up
        js = np.arange(head, j_end + up, up) if j_end != head else np.array([head])
        d = np.abs(xc[js] - xc[head])
        for h in range(1, 7):
            out[h - 1, js[d > L0 + WSTEP * h]] = 0.5
    return out


def evidence(q, t, n, days=3):
    d0 = t - (t % 288) + 60
    seen = q[d0:t].any(axis=0) if t > d0 else np.zeros(n, bool)
    for back in range(1, days + 1):
        dd = t - back * 288
        if dd - 18 >= 0:
            seen = seen | q[dd - 18:dd + 12].any(axis=0)
    return dil(seen, 2)


if __name__ == "__main__":
    W = WORK + "/t2/"; src, dst = sys.argv[1], sys.argv[2]
    probs = pickle.load(open(W + f"probs_{src}.pkl", "rb")); out = dict(probs)
    names = [str(x) for x in np.load(f"{NN}/nn_data_afdem/names.npy", allow_pickle=True)]; ci = names.index("last_queue")
    TD = os.environ.get("TFB_T2_TEST", f"{NN}/nn_test_afdem").rstrip("/"); X = np.load(f"{TD}/X.npy", mmap_mode="r"); off = np.load(f"{TD}/off.npy")
    WSEL = set(filter(None, os.environ.get("TFB_T2_WINDOWS", "").split(",")))      # 10/23: one window (t2_window_entry.py)
    pos = {str(w): i for i, w in enumerate(np.load(f"{TD}/window_id.npy", allow_pickle=True))}
    n_win = n_cells = 0; LQT = os.environ.get("TFB_T2_LQT") == "1"   # 11/03 ledger #197
    CSPL = set(os.environ.get("TFB_T2_CONE_SPLITS", "validation,private").split(","))   # 10/23: final B = the March windows only
    for split in ("validation", "private"):
        if split not in CSPL:
            continue
        for p in T2_PANELS:
            if WSEL and not any(w[3:].startswith(f"{p}_{split}_") for w in WSEL):
                continue
            P = load(p, split); ti = {s: i for i, s in enumerate(P.times)}
            for r in pd.read_csv(REL / "task2" / p / split / "window_index.csv").itertuples():
                if r.condition != "queue_ongoing" or r.window_id not in probs or (WSEL and r.window_id not in WSEL):
                    continue
                t = ti[r.forecast_origin]; Pt = asof(P, t); obs = np.isfinite(Pt.V); q = (Pt.V <= Pt.vcut[None, :]) & obs
                i = pos[r.window_id]; lq = np.asarray(X[off[i]:off[i + 1], ci], dtype=float) == 1
                if LQT:                                        # 11/03 ledger #197: the row at T where it is observed
                    vt = np.asarray(P.V[t], float); lq = np.where(np.isfinite(vt), vt <= P.vcut, lq)
                f = np.where(evidence(q, t, P.n)[None, :], 1.0, cone(P, q, obs, t, lq, p))
                out[r.window_id] = probs[r.window_id] * f; n_win += int((f < 1).any()); n_cells += int((f < 1).sum())
            load.cache_clear()
    pickle.dump(out, open(W + f"probs_{dst}.pkl", "wb"))
    print(f"cone V3: ongoing windows touched {n_win}, cells x0.5 {n_cells} -> probs_{dst}.pkl")
