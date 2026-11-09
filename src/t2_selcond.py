"""11/05 ledger #283 / #286: the ongoing probabilities conditioned on the official window selection, with the identified selector V11R
(#280b / #286: ongoing 35/40 on the training split's June official windows, the most among V6 33, V6E 29, V11RE 30, the deployed V11 16).
The official selector takes, per panel, the first origin at or after the previous official window + 360 min whose window qualifies on
the visible observation (finite speed <= vcut and eligible; coverage of T-60 .. T-5 >= 0.7; a visible history queue sum >= 2 with a link
>= 2; IoU(persistence, horizon) <= 0.9, persistence = per link the last visible observation of T-55 .. T held). The row T itself is in the
Task 1 release (finite share 0.58 / 0.60 in March / April, as the month's), and data stamped at T may be used (organizer reply 745700),
which also allows conditioning on the selection procedure and on the other official windows' origins (window_index.csv).
apply(window_id, pb) -> (pb', kept): NS = 2000 joint samples of rows T, T+5 .. T+30 (row T drawn from the T+5 probabilities; used only on
its unobserved cells): the box (the V11R persistence at T) comonotone (one uniform per sample), outside the box per row each run of cells
with p > 0.1 one uniform per sample, other cells independent. Kept samples:
  (a) the window qualifies at T: IoU(the V11R persistence at T, sample T+5 .. T+30) <= 0.9;
  (b) no earlier origin s in [max(previous official origin + 72, T-6), T-1] qualifies: for each s meeting the other conditions (coverage of
      s-12 .. s-1 >= 0.7, visible history queue sum >= 2 with a link >= 2), its horizon s+1 .. s+6 (rows < T visible with the IoU on the
      visible eligible cells; row T visible where observed, the sample elsewhere; rows > T the sample) must not hold a queue with
      IoU(V11R persistence at s = last visible observation of s-11 .. s, horizon) <= 0.9.
pb' = the kept samples' marginals (fewer than 10 kept -> pb unchanged). Deterministic: the generator is seeded by crc32(window_id).
Reads only rows <= T of the release (cachelib.asof at T+1) plus window_index.csv; no constant is fitted."""
import zlib, numpy as np, pandas as pd
from cachelib import load, asof
from common import REL

NS, H = 2000, 12
_PAN = {}


def _ffill(vq, obs, a, b):
    seen = obs[a:b]; L = b - a; last = np.where(seen.any(axis=0), L - 1 - np.argmax(seen[::-1], axis=0), -1)
    return np.where(last >= 0, vq[a:b][np.maximum(last, 0), np.arange(vq.shape[1])], False)


def _blob_uniform(q, box, rng, ns):
    n = q.shape[1]; U = rng.random((ns, q.shape[0], n))
    for r in range(q.shape[0]):
        m = (q[r] > 0.1) & ~box; j = 0
        while j < n:
            if m[j]:
                e_ = j
                while e_ + 1 < n and m[e_ + 1]:
                    e_ += 1
                U[:, r, j:e_ + 1] = rng.random((ns, 1)); j = e_ + 1
            else:
                j += 1
    return U


def condition(q, vq, obs, t, lo, rng, ns=NS):
    """q (6, n) probabilities; vq / obs (rows, n) visible queue / visible eligible, rows <= t read, rows > t unused; -> (marginals or q, kept)"""
    n = q.shape[1]; box = _ffill(vq, obs, t - H + 1, t + 1); q7 = np.vstack([q[:1], q]); u = rng.random((ns, 1, 1))
    S7 = np.where(box[None, None, :], u < q7[None], _blob_uniform(q7, box, rng, ns) < q7[None]); S = S7[:, 1:]
    inter = (box[None, None, :] & S).sum(axis=(1, 2)); uni = (box[None, None, :] | S).sum(axis=(1, 2)); keep = np.where(uni == 0, 1.0, inter / np.maximum(uni, 1)) <= 0.9
    oT = obs[t]; yT = vq[t] & oT
    for s_ in range(max(lo, t - 6, H), t):
        if obs[s_ - H:s_].mean() < 0.7:
            continue
        hq = vq[s_ - H:s_].sum(axis=0)
        if hq.sum() < 2 or hq.max() < 2:
            continue
        ps = _ffill(vq, obs, s_ - H + 1, s_ + 1); it = np.zeros(ns); un = np.zeros(ns); anyq = np.zeros(ns, bool)
        for r in range(s_ + 1, s_ + 7):
            if r <= t - 1:
                e_ = obs[r]; y_ = vq[r] & e_; p_ = ps & e_; it += (p_ & y_).sum(); un += (p_ | y_).sum(); anyq |= bool(y_.any())
            elif r == t:                                          # the observed cells of row T fixed, the rest from the sample
                p_ = ps & oT; it += (p_ & yT).sum(); un += (p_ | yT).sum(); anyq |= bool(yT.any())
                y_ = S7[:, 0] & ~oT[None]; pu = ps & ~oT; it += (pu[None] & y_).sum(axis=1); un += (pu[None] | y_).sum(axis=1); anyq |= y_.any(axis=1)
            else:
                y_ = S7[:, r - t]; it += (ps[None] & y_).sum(axis=1); un += (ps[None] | y_).sum(axis=1); anyq |= y_.any(axis=1)
        keep &= ~(anyq & (np.where(un == 0, 1.0, it / np.maximum(un, 1)) <= 0.9))
    k = int(keep.sum()); return (S[keep].mean(axis=0) if k >= 10 else q), k


def _panel(panel, split):
    if (panel, split) not in _PAN:
        P = load(panel, split); ti = {s: i for i, s in enumerate(P.times)}
        wi = pd.read_csv(REL / "task2" / panel / split / "window_index.csv"); ts = {w: ti[o] for w, o in zip(wi.window_id, wi.forecast_origin)}
        prev = {w: max([u for u in ts.values() if u < t], default=-10 ** 9) for w, t in ts.items()}
        _PAN[(panel, split)] = (P, ts, prev)
    return _PAN[(panel, split)]


def apply(window_id, pb, panel=None, split=None):
    if panel is None:                                            # Q2_<panel>_<split>_<nnn>
        body = window_id[3:]; split = "validation" if "_validation_" in body else "private" if "_private_" in body else "train"; panel = body.split(f"_{split}_")[0]
    P, ts, prev = _panel(panel, split); t = ts[window_id]; Pt = asof(P, t + 1)   # rows <= T (the row T is released, 745700)
    fin = np.isfinite(Pt.V); vq = fin & (np.nan_to_num(Pt.V, nan=1e9) <= Pt.vcut[None, :]) & Pt.E; obs = fin & Pt.E
    return condition(pb, vq, obs, t, prev[window_id] + 72, np.random.default_rng(zlib.crc32(window_id.encode())))
