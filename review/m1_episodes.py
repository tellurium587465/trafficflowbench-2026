"""M1 (memorization audit 10/09): queue episodes of the train months and their 'stationary vs incident' descriptors.

Episode = connected component of queued cells (v <= 0.6 vf_links, eligible) after a 3x3 dilation, on (time x link).
Per episode (all computed from train months only):
  head0      most downstream queued link, median over the first 30 min
  r_cap      discharge ratio: flow at head+1 (free-flowing) / fd capacity of head+1, median over the episode
  r_typ      flow at head+1 / typical flow at that link and 5-min slot (median over other months, same day type)
  r_site     the site's own usual r_cap (median over episodes at head0+-1 in other months), NaN if < 3 episodes
  f_site     daily frequency of a queue at head0+-1 in the 30-min bin of the start (other months, same day type)
  drop10     v/vf at head0 10 min before start minus at start+10 min (sharpness of the onset)
usage: python m1_episodes.py  -> m1_episodes.pkl (pandas)
"""
import sys, numpy as np, pandas as pd
from scipy import ndimage
sys.path.insert(0, r"C:/Users/user/tfb/src")
from cachelib import load, T2_PANELS
from t2_windows import queue_matrix

rows = []
for p in T2_PANELS:
    P = load(p, "train"); n = P.n; T = P.T; D = T // 288
    Qm = queue_matrix(P)
    Q = np.where(np.isfinite(P.Q), P.Q, np.nan).astype(np.float32)
    vr = (P.V / P.vf[None, :]).astype(np.float32)
    days = [str(x) for x in P.day]; mon = np.array([d[:7] for d in days]); wkend = np.array([pd.Timestamp(d).dayofweek >= 5 for d in days])
    months = sorted(set(mon))
    Q3 = Q.reshape(D, 288, n); Qd = Qm.reshape(D, 288, n)
    # typical flow per (held-out month, day type, slot, link) and queue frequency per 30-min bin
    typ = {}; frq = {}
    qb = Qd.reshape(D, 48, 6, n).any(axis=2)                   # queued in the 30-min bin (D, 48, n)
    for m in months:
        for we in (False, True):
            sel = (mon != m) & (wkend == we)
            typ[(m, we)] = np.nanmedian(Q3[sel], axis=0)     # (288, n)
            frq[(m, we)] = qb[sel].mean(axis=0)              # (48, n)
    lab, nl = ndimage.label(ndimage.binary_dilation(Qm, np.ones((3, 3), bool)) & True)
    lab = np.where(Qm, lab, 0)
    objs = ndimage.find_objects(lab)
    for i, sl in enumerate(objs):
        if sl is None:
            continue
        sub = lab[sl] == i + 1
        t0, l0 = sl[0].start, sl[1].start
        nrow = sub.shape[0]
        if sub.sum() < 2:
            continue
        has = sub.any(axis=1)
        head = np.where(has, l0 + (sub.shape[1] - 1 - np.argmax(sub[:, ::-1], axis=1)), -1)
        tail = np.where(has, l0 + np.argmax(sub, axis=1), -1)
        h0 = int(np.median(head[:6][head[:6] >= 0]))
        d = t0 // 288; m = mon[d]; we = wkend[d]; slot = t0 % 288
        rc, rt = [], []
        for r in range(min(nrow, 36)):
            h = head[r]; j = h + 1
            if h < 0 or j >= n or Qm[t0 + r, j] or not np.isfinite(Q[t0 + r, j]):
                continue
            rc.append(Q[t0 + r, j] / P.cap[j])
            ty = typ[(m, we)][(t0 + r) % 288, j]
            if np.isfinite(ty) and ty > 0:
                rt.append(Q[t0 + r, j] / ty)
        fs = frq[(m, we)][slot // 6, max(0, h0 - 1):h0 + 2].max()
        a, b = max(t0 - 2, 0), min(t0 + 2, T - 1)
        drop = np.nanmean(vr[max(t0 - 3, 0):t0, h0]) - np.nanmean(vr[t0:t0 + 3, h0]) if t0 >= 3 else np.nan
        rows.append(dict(panel=p, t0=int(t0), t1=int(t0 + nrow - 1), dur=nrow, cells=int(sub.sum()), date=days[d], month=str(m),
                         weekend=bool(we), tod=slot / 12, head0=h0, span=int(head.max() - tail[tail >= 0].min() + 1),
                         r_cap=np.median(rc) if rc else np.nan, r_typ=np.median(rt) if rt else np.nan, n_r=len(rc),
                         f_site=float(fs), drop10=float(drop)))
    print(p, "episodes", sum(1 for r in rows if r["panel"] == p), flush=True)

E = pd.DataFrame(rows)
# r_site: the site's own usual discharge ratio in other months (recurrent sites with hidden constraints sit at 0.6-0.7)
E["r_site"] = np.nan
for (p, m), g in E.groupby(["panel", "month"]):
    o = E[(E.panel == p) & (E.month != m) & (E.dur >= 6) & E.r_cap.notna()]
    for i, r in g.iterrows():
        s = o[(o.head0 - r.head0).abs() <= 1].r_cap
        E.at[i, "r_site"] = s.median() if len(s) >= 3 else np.nan
E.to_pickle(r"C:/Users/user/tfb/review/m1_episodes.pkl")
print(E.describe().T.to_string())
