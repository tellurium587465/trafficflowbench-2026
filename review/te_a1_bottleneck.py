"""TE-A1 (traffic-engineering review): bottleneck physics in the generator.
For each split: congested cells C = V <= vcut. Per time step, congested segments (gaps <= 1 link bridged);
head = most downstream congested link, bottleneck b = head+1.
Reports per split:
  - where heads sit (top locations, share), structural cause at b (lane drop / cap drop vs head / on-ramp)
  - discharge ratio Q[b]/cap[b] and Q[head]/cap[b] during stable queues (capacity drop?)
  - pre-breakdown max flow over 30 min before onset at b, / cap[b]
  - head movement within an episode
Usage: python te_a1_bottleneck.py PANEL [PANEL...]"""
import sys, numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, r"C:/Users/user/tfb/src")
from cachelib import load


def segments(row):
    idx = np.flatnonzero(row)
    if idx.size == 0:
        return []
    out = []; s = idx[0]; p = idx[0]
    for i in idx[1:]:
        if i - p > 2:
            out.append((s, p)); s = i
        p = i
    out.append((s, p))
    return out


def analyse(panel, split):
    P = load(panel, split)
    V, Q = P.V, P.Q
    if split == "train":
        V = V.copy(); Q = Q.copy(); V[P.blank] = np.nan; Q[P.blank] = np.nan   # what a participant sees
    fin = np.isfinite(V)
    C = fin & (V <= P.vcut[None, :])
    n, T = P.n, P.T
    rows = []
    for t in range(T):
        for (a, h) in segments(C[t]):
            if h - a < 1 or h >= n - 1:
                continue
            rows.append((t, a, h))
    d = pd.DataFrame(rows, columns=["t", "tail", "head"])
    if d.empty:
        return None
    b = d["head"].to_numpy() + 1
    t = d.t.to_numpy()
    d["qb"] = Q[t, b] / P.cap[b]
    d["qh_capb"] = Q[t, d["head"].to_numpy()] / P.cap[b]
    d["qb1"] = np.where(b + 1 < n, Q[t, np.minimum(b + 1, n - 1)] / P.cap[b], np.nan)
    d["vb"] = V[t, b] / P.vf[b]
    # stable: same head at t-1 and t+1
    key = set(zip(d.t, d["head"]))
    d["stable"] = [((tt - 1, hh) in key) and ((tt + 1, hh) in key) for tt, hh in zip(d.t, d["head"])]
    # structure at b
    lanes, cap = P.lanes, P.cap
    hb = d["head"].to_numpy()
    d["lane_drop"] = lanes[b] < lanes[hb]
    d["cap_drop"] = cap[b] < cap[hb] - 1
    d["onramp_b"] = P.on_attached[b] | P.on_attached[hb]
    return P, d, Q, C


def onset_events(P, d, Q, C):
    """Head h first congested at t0 with no congestion within +-3 links in the previous 12 steps."""
    ev = []
    n = P.n
    for (t0, h) in zip(d.t.to_numpy(), d["head"].to_numpy()):
        if t0 < 13:
            continue
        lo, hi = max(h - 3, 0), min(h + 4, n)
        if C[t0 - 12:t0, lo:hi].any():
            continue
        b = h + 1
        pre = Q[t0 - 6:t0, b] / P.cap[b]
        pre_h = Q[t0 - 6:t0, h] / P.cap[b]
        post = Q[t0 + 2:t0 + 12, b] / P.cap[b]
        ev.append(dict(t0=t0, head=h, pre_max_b=np.nanmax(pre) if np.isfinite(pre).any() else np.nan,
                       pre_max_h=np.nanmax(pre_h) if np.isfinite(pre_h).any() else np.nan,
                       post_med_b=np.nanmedian(post) if np.isfinite(post).any() else np.nan,
                       tod=(t0 % 288) / 12.0))
    return pd.DataFrame(ev).drop_duplicates(["t0", "head"])


if __name__ == "__main__":
    pd.set_option("display.width", 220)
    for panel in sys.argv[1:]:
        for split in ("train", "validation", "private"):
            r = analyse(panel, split)
            if r is None:
                print(panel, split, "no queues"); continue
            P, d, Q, C = r
            st = d[d.stable]
            print(f"\n=== {panel} {split}: queue-steps {len(d)}  stable {len(st)}  n_links {P.n}")
            top = d["head"].value_counts(normalize=True).head(6)
            print(" head share (top6):", ", ".join(f"{P.links[h]}(i{h}):{s:.2f}" for h, s in top.items()))
            print(" structure at b (share of queue-steps): lane_drop %.2f cap_drop %.2f onramp %.2f" % (
                d.lane_drop.mean(), d.cap_drop.mean(), d.onramp_b.mean()))
            g = st.groupby("head").agg(steps=("t", "size"), qb_med=("qb", "median"), qh_med=("qh_capb", "median"),
                                       qb1_med=("qb1", "median"), vb_med=("vb", "median"), q10=("qb", lambda x: np.nanpercentile(x, 10)),
                                       q90=("qb", lambda x: np.nanpercentile(x, 90)))
            g = g.sort_values("steps", ascending=False).head(8)
            g.index = [f"{P.links[h]}(i{h}) cap_b={P.cap[h+1]:.0f} L{P.lanes[h]:.0f}->{P.lanes[h+1]:.0f} capL {P.cap[h]/P.lanes[h]:.0f}->{P.cap[h+1]/P.lanes[h+1]:.0f}" for h in g.index]
            print(" stable-queue discharge by head (Q/cap_b):"); print(g.round(3).to_string())
            ev = onset_events(P, d, Q, C)
            if len(ev):
                print(f" onset events {len(ev)}: pre-max Q[b]/cap_b median {ev.pre_max_b.median():.3f} (p10 {ev.pre_max_b.quantile(.1):.3f}, p90 {ev.pre_max_b.quantile(.9):.3f});"
                      f" post median {ev.post_med_b.median():.3f}; drop ratio post/pre median {(ev.post_med_b/ev.pre_max_b).median():.3f}")
            load.cache_clear()
