"""E3a on M3 (pre-registered in STATE.md 10/22 10:50): evidence that looks past an earlier day's blank.
On each evidence day, if the slot window (onset: rows dd-6..dd+11, ongoing: dd-18..dd+11) overlaps a run of >= 18 rows
in which every link is missing (an official window's blank; in M3 also the trap-4 hidden rows), the 18 rows right after
that run are read as well. Everything else as the production rule (x0.25 onset / x0.85 ongoing, +-2 / +-1 links, at most
3 earlier days, k-day limit, trap 4) through t2_m3_gate's functions; the far gate is unchanged.
Compared with the current rule on the same windows and blends (q24f onset and ongoing; seeds 0-3), cluster bootstrap.
usage: python t2_m3_e3a.py"""
import sys
sys.argv = [sys.argv[0], "0,1,2,3"]
import numpy as np, pandas as pd
import t2_m3_gate as G

ALLNAN = {}


def allnan(p):
    if p not in ALLNAN:
        P, _ = G.panel(p); ALLNAN[p] = ~np.isfinite(P.V).any(axis=1)
    return ALLNAN[p]


def seen_rows_e3a(p, t, k, lo, hi, cap=3):
    P, q = G.panel(p); an = allnan(p).copy(); hid = np.zeros(P.T, bool)
    s = np.zeros(P.n, bool); nd = 0; n_look = 0
    for back in range(1, min(cap, k) + 1):
        dd = t - back * 288
        if dd + lo < 0:
            break
        for e in G.EVENTS[p][np.abs(G.EVENTS[p] - dd) <= 12]:         # trap 4 rows count as blank (missing) rows
            hid[e + 1:e + 19] = True
        miss = an | hid
        a, b = dd + lo, dd + hi
        blk = q[a:b] & ~hid[a:b, None]
        s |= blk.any(axis=0); nd += 1
        r = a                                                         # runs of >= 18 missing rows overlapping [a, b)
        while r < b:
            if miss[r]:
                rs = r
                while rs > 0 and miss[rs - 1]:
                    rs -= 1
                re = r
                while re < P.T and miss[re]:
                    re += 1
                if re - rs >= 18:
                    look = q[re:re + 18] & ~hid[re:re + 18, None]
                    s |= look.any(axis=0); n_look += 1
                r = re
            else:
                r += 1
    return s, nd, n_look


def onset_e3a(p_, m):
    pb = p_.copy(); s, nd, nl = seen_rows_e3a(m["panel"], m["t"], m["k"], -6, 12)
    if nd:
        pb[-1] = pb[-1] * np.where(G.dil(s, 2), 1.0, 0.25)
    return G.best_set(pb.ravel(), np.ones(pb.size, bool)).reshape(pb.shape), nl


def ongoing_e3a(p_, m):
    s, nd, nl = seen_rows_e3a(m["panel"], m["t"], m["k"], -18, 12)
    pb = p_ * np.where(G.dil(s, 1), 1.0, 0.85)[None, :] if nd else p_.copy()
    sel = G.best_set(pb.ravel(), np.ones(pb.size, bool)).reshape(pb.shape)
    gated = G.ongoing_pred(p_, m, True)                               # the gate's own decision on the current rule
    base = G.ongoing_pred(p_, m, False)
    return sel & ~(base & ~gated), nl                                 # same cells removed by the far gate as production


rows = []
for w in G.WID:
    m = G.META[w]; r = dict(cond=m["cond"], cl=f"{m['panel']}_{m['day']}", band="0.5-0.6" if m["cov"] < 0.6 else "0.6-0.7")
    if m["cond"] == "queue_onset":
        r["cur"] = G.iou(G.onset_pred(G.NOMEM[w], m, "cur"), m); sel, nl = onset_e3a(G.NOMEM[w], m); r["e3a"] = G.iou(sel, m)
    else:
        r["cur"] = G.iou(G.ongoing_pred(G.NOMEM[w], m, True), m); sel, nl = ongoing_e3a(G.NOMEM[w], m); r["e3a"] = G.iou(sel, m)
    r["looked"] = nl; rows.append(r)
D = pd.DataFrame(rows); rng = np.random.default_rng(0)
for cond, g in D.groupby("cond"):
    u = g.cl.unique(); idx = {x: np.flatnonzero(g.cl.to_numpy() == x) for x in u}; d = (g.e3a - g.cur).to_numpy()
    se = np.std([d[np.concatenate([idx[x] for x in rng.choice(u, len(u))])].mean() for _ in range(1000)])
    print(f"{cond}: windows {len(g)}, with a look past a blank {int((g.looked > 0).sum())}, changed {int((d != 0).sum())}, "
          f"cur {g.cur.mean():.4f} e3a {g.e3a.mean():.4f} diff {d.mean():+.4f} (SE {se:.4f}, {d.mean() / max(se, 1e-9):+.1f} SE)", flush=True)
for (cond, band), g in D.groupby(["cond", "band"]):
    print(f"  {cond} band {band}: windows {len(g)}, diff {(g.e3a - g.cur).mean():+.4f}")
