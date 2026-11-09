"""G161 (11/01, ledger #117c, follow-up of #117): does the final Task 1 output follow the month's free-flow level of each link? The harnesses are training months, where x = 1, so a miss
could not have shown there. Per link and month: h = median of the output on hidden cells / median of the observed free cells, both in rows with no observed queue on the panel within
+-6 rows and on links with >= 200 such hidden cells. If the output ignored the month's level, h would be 1 / x; if it follows, h = 1 whatever x is.
Reads only our own output and the released layer. usage: py -B -X utf8 g161_t1_free_level.py"""
import sys
sys.dont_write_bytecode = True; sys.path.insert(0, r"C:/Users/user/tfb/src")
import numpy as np, pandas as pd
from cachelib import load, PANELS
from common import WORK

S = pd.read_parquet(WORK + "/t1/state_f3ens3c_ad2_gapml_clip_fd_bnd.parquet"); print(S.columns.tolist(), len(S), flush=True)
rows = []
from common import REL
LV = {}


def level(p):
    if p not in LV:
        Pt = load(p, "train"); V = np.where(Pt.E, Pt.V, np.nan); q = (V <= Pt.vcut[None, :]).any(axis=1); nq = np.convolve(q.astype(int), np.ones(13, int), "same") == 0
        with np.errstate(all="ignore"):
            LV[p] = np.nanmedian(V[nq], axis=0)
    return LV[p]


sc = [c for c in S.columns if "speed" in c.lower()][0]; lc = [c for c in S.columns if "link" in c.lower()][0]; tc = [c for c in S.columns if "time" in c.lower()][0]; pc = [c for c in S.columns if "panel" in c.lower()]
for p, g in (S.groupby(pc[0]) if pc else [("", S)]):
    for sp in ("validation", "private"):
        try:
            P = load(p, sp)
        except Exception:
            continue
        ti = {str(t)[:16]: i for i, t in enumerate(P.times)}; li = {l: i for i, l in enumerate(P.links)}; r_ = g[tc].astype(str).str[:16].map(ti); c_ = g[lc].astype(str).map(li); ok = r_.notna() & c_.notna()
        if not ok.any():
            continue
        Hat = np.full((P.T, P.n), np.nan); Hat[r_[ok].astype(int).to_numpy(), c_[ok].astype(int).to_numpy()] = g.loc[ok, sc].to_numpy(float)
        V = np.where(P.E, P.V, np.nan); q = (V <= P.vcut[None, :]).any(axis=1); nq = np.convolve(q.astype(int), np.ones(13, int), "same") == 0; hid = np.isfinite(Hat) & ~np.isfinite(P.V) & P.E & nq[:, None]; vis = np.isfinite(V) & nq[:, None]
        with np.errstate(all="ignore"):
            mh = np.nanmedian(np.where(hid, Hat, np.nan), axis=0); mv = np.nanmedian(np.where(vis, V, np.nan), axis=0)
        n = hid.sum(axis=0); x = mv / level(p)
        rows.append(pd.DataFrame(dict(panel=p, split=sp, link=np.arange(P.n), n=n, h=mh / mv, x=x)))
D = pd.concat(rows, ignore_index=True); D = D[(D.n >= 200) & np.isfinite(D.h) & np.isfinite(D.x)]; D.to_csv(r"C:/Users/user/tfb/work/t1/g161_free_level.csv", index=False)
for sp, d in D.groupby("split"):
    b = np.polyfit(np.log(d.x), np.log(d.h), 1)[0]
    print(f"{sp}: links {len(d)} | h median {d.h.median():.5f}, sd {d.h.std():.5f}, 1-99% {d.h.quantile(.01):.4f}..{d.h.quantile(.99):.4f} | sd of x {d.x.std():.5f} | slope log h on log x {b:+.3f} (0 = follows, -1 = ignores) | corr {np.corrcoef(np.log(d.x), np.log(d.h))[0, 1]:+.3f}")
    for lo, hi in ((0, .98), (.98, .99), (.99, 1.01), (1.01, 1.02), (1.02, 9)):
        e = d[(d.x >= lo) & (d.x < hi)]; print(f"   x {lo}-{hi}: links {len(e)}, h median {e.h.median():.5f}")
print("km/h at 117: sd of the level miss", round(float(D.h.std() * 117), 3))
