"""TE-A8: do the per-split Task 4 link counts (c = A f*, the month's OD loading of each link) explain WHERE the month's
queues are? The OD (f*) is drawn independently per month, so the flow-to-capacity profile along the corridor moves.
Per panel and split: c/cap per link; congestion share per link (eligible released cells with v <= vcut);
'first queue of the day' link per day (the onset-like event).
Reports: corr(c/cap, congestion share) within split; corr of the CHANGE (val/priv minus train) in c/cap vs change in
congestion share; and how often the day's first queue sits at a link whose c/cap is in the corridor's top decile."""
import sys, numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, r"C:/Users/user/tfb/src")
from cachelib import load, T2_PANELS
from common import REL


def stats(panel, split):
    P = load(panel, split)
    V = P.V
    if split == "train":
        V = V.copy(); V[P.blank] = np.nan
    ok = np.isfinite(V)
    C = ok & (V <= P.vcut[None, :])
    cong = C.sum(0) / np.maximum(ok.sum(0), 1)
    c = pd.read_csv(REL / "task4" / panel / split / "synthetic_link_counts.csv", dtype={"link_id": str}).set_index("link_id")["count"]
    cc = c.reindex(P.links).to_numpy(float) / P.cap
    first = []
    for d0 in range(0, P.T, 288):
        r = np.flatnonzero(C[d0:d0 + 288].any(axis=1))
        if len(r):
            first.append(np.flatnonzero(C[d0 + r[0]]).max())     # most downstream congested link at the first queue time
    load.cache_clear()
    return cc, cong, np.array(first, int)


rows = []
for p in T2_PANELS:
    S = {sp: stats(p, sp) for sp in ("train", "validation", "private")}
    cct, cgt, _ = S["train"]
    for sp, (cc, cg, first) in S.items():
        ok = np.isfinite(cc)
        top = cc >= np.nanpercentile(cc, 90)
        near_top = np.array([top[max(f - 2, 0):f + 3].any() for f in first]) if len(first) else np.array([])
        r = dict(panel=p, split=sp, corr_level=np.corrcoef(cc[ok], cg[ok])[0, 1], first_near_top10pct=near_top.mean() if len(first) else np.nan,
                 cc_max=np.nanmax(cc), sd_dcc=np.nanstd(cc - cct) if sp != "train" else 0.0)
        if sp != "train":
            dc = cc - cct; dg = cg - cgt; m = np.isfinite(dc) & np.isfinite(dg)
            r["corr_change"] = np.corrcoef(dc[m], dg[m])[0, 1]
        rows.append(r)
d = pd.DataFrame(rows)
pd.set_option("display.width", 200)
print(d.round(3).to_string(index=False))
print(d.groupby("split")[["corr_level", "first_near_top10pct", "corr_change", "sd_dcc"]].mean().round(3))
