"""Audit A7: what moved the bottlenecks in validation/private?
Per link and month: free-flow speed level vmed = median(v/vf | v > 0.9 vf), flow level (median q/cap in free flow),
high flow q99 = 99th pct of q/cap, congestion rate = share of eligible released cells with v <= v_cut.
Train months: unmasked layer minus Task 1 blanks (= what a participant sees); val/private: masked layer.
Question: does the per-link change in vmed / q99 (val/private vs train mean) explain the per-link change in congestion?"""
import sys, numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, r"C:/Users/user/tfb/src")
from cachelib import load, T2_PANELS


def month_stats(P, V, Q, E, months):
    out = {}
    for m in np.unique(months):
        s = months == m
        v, q, e = V[s], Q[s], E[s]
        ok = e & np.isfinite(v)
        ff = ok & (v > 0.9 * P.vf[None, :])
        vr = np.where(ff, v / P.vf[None, :], np.nan); qr = np.where(ok, q / P.cap[None, :], np.nan)
        out[m] = dict(vmed=np.nanmedian(vr, axis=0), q99=np.nanpercentile(qr, 99, axis=0),
                      cong=((v <= P.vcut[None, :]) & ok).sum(0) / np.maximum(ok.sum(0), 1),
                      qff=np.nanmedian(np.where(ff, q / P.cap[None, :], np.nan), axis=0))
    return out


rows = []
for p in T2_PANELS:
    P = load(p, "train")
    V = P.V.copy(); Q = P.Q.copy(); V[P.blank] = np.nan; Q[P.blank] = np.nan
    tr = month_stats(P, V, Q, P.E, np.array([s[:7] for s in P.times]))
    del V, Q
    load.cache_clear()
    te = {}
    for sp in ("validation", "private"):
        Pt = load(p, sp); te.update(month_stats(Pt, Pt.V, Pt.Q, Pt.E, np.array([s[:7] for s in Pt.times]))); load.cache_clear()
    keys = sorted(tr)
    base = {k: np.nanmean([tr[m][k] for m in keys], axis=0) for k in ("vmed", "q99", "cong", "qff")}
    for m, st in list(tr.items()) + list(te.items()):
        dv = st["vmed"] - base["vmed"]; dq = st["q99"] - base["q99"]; dc = st["cong"] - base["cong"]; dff = st["qff"] - base["qff"]
        ok = np.isfinite(dv) & np.isfinite(dq) & np.isfinite(dc)
        cor = lambda a, b: float(np.corrcoef(a[ok], b[ok])[0, 1]) if ok.sum() > 5 else np.nan
        rows.append(dict(panel=p, month=m, split="train" if m in tr else ("val" if m == "2031-03" else "priv"),
                         sd_dvmed_pct=100 * np.nanstd(dv), sd_dq99=np.nanstd(dq), sd_dqff=np.nanstd(dff), tv_cong=0.5 * np.abs(st["cong"] / max(st["cong"].sum(), 1e-9) - base["cong"] / max(base["cong"].sum(), 1e-9)).sum(),
                         corr_dcong_dvmed=cor(dc, dv), corr_dcong_dq99=cor(dc, dq), corr_dvmed_dqff=cor(dv, dff)))
d = pd.DataFrame(rows)
pd.set_option("display.width", 220)
print(d.groupby("split")[["sd_dvmed_pct", "sd_dq99", "sd_dqff", "tv_cong", "corr_dcong_dvmed", "corr_dcong_dq99", "corr_dvmed_dqff"]].mean().round(4))
print(d[d.split != "train"].round(4).to_string())
