"""TE-A5: physical direction check + shock (tail) kinematics.
(1) For isolated queue births (link j becomes congested, nothing congested within +-4 links in the previous hour),
    the segment containing j 30 min later: how far did it extend to lower index (tail side if flow is low->high) and higher index?
(2) Concentration of segment max-index (head) vs min-index (tail) positions.
(3) Tail kinematics for ongoing queues: Rankine-Hugoniot prediction of the tail displacement over 30 min vs actual,
    vs persistence and vs linear extrapolation of the last 30 min.
Usage: python te_a5_queue_growth.py PANEL [PANEL...]"""
import sys, numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, r"C:/Users/user/tfb/src")
from cachelib import load
from te_a1_bottleneck import segments


def seg_of(row, j):
    for a, b in segments(row):
        if a <= j <= b:
            return a, b
    return None


for panel in sys.argv[1:]:
    P = load(panel, "train")
    V, Q = P.V, P.Q
    C = np.isfinite(V) & (V <= P.vcut[None, :])
    x0 = np.concatenate([[0.0], np.cumsum(P.length)])      # upstream edge of link i = x0[i] (low index = start)
    n, T = P.n, P.T
    ext_lo, ext_hi, births = [], [], 0
    for t in range(12, T - 7):
        newc = np.flatnonzero(C[t] & ~C[t - 1])
        for j in newc:
            lo, hi = max(j - 4, 0), min(j + 5, n)
            if C[t - 12:t, lo:hi].any() or C[t, lo:hi].sum() > 1:
                continue
            s = seg_of(C[t + 6], j)
            if s is None:
                continue
            births += 1
            ext_lo.append(x0[j] - x0[s[0]]); ext_hi.append(x0[s[1] + 1] - x0[j + 1])
    ext_lo, ext_hi = np.array(ext_lo), np.array(ext_hi)
    print(f"\n=== {panel}: isolated births {births}; 30 min later, extension toward LOWER index: mean {ext_lo.mean():.2f} km "
          f"(>0.5km on {np.mean(ext_lo > 0.5):.2f}); toward HIGHER index: mean {ext_hi.mean():.2f} km (>0.5km on {np.mean(ext_hi > 0.5):.2f})")
    # (2) concentration
    heads, tails = [], []
    for t in range(0, T, 3):
        for a, b in segments(C[t]):
            if b - a >= 1:
                heads.append(b); tails.append(a)
    hs = pd.Series(heads).value_counts(normalize=True); ts = pd.Series(tails).value_counts(normalize=True)
    print(f" segment max-index top3 share {hs.head(3).sum():.2f}, min-index top3 share {ts.head(3).sum():.2f}; "
          f"distinct max {len(hs)}, distinct min {len(ts)}")
    # (3) tail kinematics: segments with length >= 2 links at t and the 'same' queue at t+6 (contains old head +-1)
    kq = Q / np.maximum(V, 1.0)
    rows = []
    for t in range(6, T - 6, 2):
        for a, b in segments(C[t]):
            if b - a < 1 or a < 4:
                continue
            s6 = None
            for a2, b2 in segments(C[t + 6]):
                if a2 <= b and b2 >= b - 1:
                    s6 = (a2, b2); break
            s_6 = None
            for a2, b2 in segments(C[t - 6]):
                if a2 <= b and b2 >= b - 1:
                    s_6 = (a2, b2); break
            up = slice(a - 3, a); qz = slice(a, min(a + 3, b + 1))
            qu, ku = np.nanmean(Q[t, up]), np.nanmean(kq[t, up])
            qq, kk = np.nanmean(Q[t, qz]), np.nanmean(kq[t, qz])
            if not np.isfinite([qu, ku, qq, kk]).all() or abs(ku - kk) < 1e-6:
                continue
            # per-lane normalisation not needed: speed = dq/dk in total units along the corridor (lanes vary -> use per-lane)
            s = (qu - qq) / (ku - kk)                               # km/h (vehicle conservation), negative = upstream
            pred = np.clip(s * 0.5, -15, 15)                        # km in 30 min
            xt = x0[a]
            act = (x0[s6[0]] - xt) if s6 is not None else np.nan    # NaN = queue gone
            lin = (xt - x0[s_6[0]]) if s_6 is not None else np.nan
            rows.append(dict(t=t, a=a, b=b, s=s, pred=pred, act=act, lin=lin, gone=s6 is None,
                             headmove=(x0[s6[1] + 1] - x0[b + 1]) if s6 is not None else np.nan))
    d = pd.DataFrame(rows)
    ok = d.dropna(subset=["act"])
    ok2 = ok.dropna(subset=["lin"])
    mae = lambda e: float(np.mean(np.abs(e)))
    print(f" tail kinematics: n={len(ok)} (gone in 30 min: {d.gone.mean():.2f}); actual tail shift km: mean {ok.act.mean():+.2f}, |.| {mae(ok.act):.2f}")
    print(f"   MAE persistence {mae(ok.act):.2f} | RH {mae(ok.act - ok.pred):.2f} | RH x0.5 {mae(ok.act - 0.5 * ok.pred):.2f} | "
          f"linear(last 30) {mae(ok2.act - ok2.lin):.2f} vs persistence on same {mae(ok2.act):.2f} | corr(RH, act) {np.corrcoef(ok.pred, ok.act)[0, 1]:.2f}")
    print(f"   RH speed s km/h: p10/50/90 {np.round(np.nanpercentile(d.s, [10, 50, 90]), 1)} ; w median {np.median(P.w):.1f} km/h")
    print(f"   head shift (km) over 30 min: |.|<0.3 km on {np.mean(np.abs(ok.headmove) < 0.3):.2f}, mean {ok.headmove.mean():+.2f}")
    load.cache_clear()
