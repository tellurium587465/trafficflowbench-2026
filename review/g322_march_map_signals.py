"""G322 (11/05, ledger #257, measurement only): can March's new map be read from per-link changes visible in the data? Per link of the eight
Task 2 panels, March (validation split, released masked data) against the training months (train split): (a) free-flow speed change =
median speed over the visible free rows (v > vcut) in March / the same in training, minus 1; (b) the 99.5 % flow quantile over visible rows,
March / training, minus 1; (c) the head rate log ratio = log((onsets per visible row in March + eps) / (the same in training + eps)), with a
link-level onset as G314 (queued after >= 12 visible free rows) and eps = 0.5 onset over the month's visible rows of the link (half an event).
Spearman of (c) with (a) and (b), per panel and overall. Reading (#257): overall |rho| >= 0.2 and the same sign in most panels.
usage: py -B -X utf8 g322_march_map_signals.py"""
import sys, numpy as np, pandas as pd
sys.path.insert(0, r"C:/Users/user/tfb_a8/src")
from cachelib import load, T2_PANELS
from scipy.stats import spearmanr


def stats(P):
    V, Q = P.V, P.Q; fin = np.isfinite(V); qd = fin & (V <= P.vcut[None, :]); fr = fin & ~qd
    free12 = np.zeros_like(fr); run = np.zeros(P.n, int)
    for r in range(P.T):
        free12[r] = run >= 12; run = np.where(fr[r], run + 1, np.where(fin[r], 0, run))
    ons = (qd & free12).sum(axis=0); nvis = fin.sum(axis=0)
    vff = np.array([np.median(V[fr[:, j], j]) if fr[:, j].any() else np.nan for j in range(P.n)])
    q995 = np.array([np.nanquantile(np.where(fin[:, j], Q[:, j], np.nan), 0.995) if fin[:, j].any() else np.nan for j in range(P.n)])
    return ons, nvis, vff, q995


rows = []
for p in T2_PANELS:
    Pt = load(p, "train"); ot, nt, vt, qt = stats(Pt); load.cache_clear(); Pm = load(p, "validation"); om, nm, vm, qm = stats(Pm); load.cache_clear()
    rt = (ot + 0.5) / np.maximum(nt, 1); rm = (om + 0.5) / np.maximum(nm, 1)
    for j in range(len(ot)):
        rows.append(dict(panel=p, link=j, a=vm[j] / vt[j] - 1, b=qm[j] / qt[j] - 1, c=np.log(rm[j] / rt[j]), ons_t=int(ot[j]), ons_m=int(om[j])))
D = pd.DataFrame(rows).replace([np.inf, -np.inf], np.nan).dropna(subset=["a", "b", "c"]); D.to_csv("C:/Users/user/tfb/work/t2/g322_links.csv", index=False)
out = []
for p, d in D.groupby("panel"):
    ra = spearmanr(d.c, d.a)[0]; rb = spearmanr(d.c, d.b)[0]; out.append((p, len(d), ra, rb)); print(f"{p}: links {len(d)}, rho(c, a) {ra:+.3f}, rho(c, b) {rb:+.3f} | a median {d.a.median():+.4f}, b median {d.b.median():+.4f}", flush=True)
ra = spearmanr(D.c, D.a)[0]; rb = spearmanr(D.c, D.b)[0]
for nm, r, k in (("a (free-flow speed)", ra, 2), ("b (99.5 % flow)", rb, 3)):
    same = sum(np.sign(x[k]) == np.sign(r) for x in out)
    print(f"overall rho(c, {nm}) {r:+.3f}, panels with the same sign {same}/{len(out)} | reading (|rho| >= 0.2 and most panels): {abs(r) >= 0.2 and same > len(out) / 2}", flush=True)
print(f"links {len(D)}; onsets per link: training median {D.ons_t.median():.0f}, March median {D.ons_m.median():.0f}", flush=True)
