"""G341 (11/05, ledger #272, measurement only): the shift of the demand's time-of-day shape. Per panel x link, weekday rows only, flow Q
(released data, NaN-aware): the hourly profile (24 bins, mean Q) divided by its mean = the shape. Per month m: the correlation of m's shape
with the pool of the training months (the training months themselves leave themselves out of the pool), the mean |shape_m - shape_pool|,
and the shift of the peak hour (circular, hours). March (validation) and April (private) against the training months' month-to-month spread.
Also: March rare births per link (head onsets as G336: queued after >= 12 free rows, downstream neighbour free; rarity_table at that link <= 0.1)
against the link's March shape distance, Spearman. usage: py -B -X utf8 g341_demand_shape.py"""
import sys
import numpy as np, pandas as pd
sys.path.insert(0, r"C:/Users/user/tfb_a8/src"); sys.path.insert(0, r"C:/Users/user/tfb/src")
from cachelib import load, T2_PANELS
import t2_phys_onset as PO
from scipy.stats import spearmanr
RT = PO.rarity_table(); rows = []; rare_rows = []


def shape(Q, times):
    wd = np.array([pd.Timestamp(s[:10]).dayofweek < 5 for s in times]); hr = np.array([int(s[11:13]) for s in times]); out = np.full((24, Q.shape[1]), np.nan)
    for h in range(24):
        m = wd & (hr == h)
        if m.any():
            out[h] = np.nanmean(Q[m], axis=0)
    return out / np.nanmean(out, axis=0, keepdims=True)


def heads(V, vcut):
    fin = np.isfinite(V); qd = fin & (V <= vcut[None, :]); fr = fin & ~qd; free12 = np.zeros_like(fr); run = np.zeros(V.shape[1], int)
    for r in range(len(V)):
        free12[r] = run >= 12; run = np.where(fr[r], run + 1, np.where(fin[r], 0, run))
    return qd & free12 & ~np.concatenate([qd[:, 1:], np.zeros((len(qd), 1), bool)], axis=1)


def cmp(a, b):
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 12:
        return np.nan, np.nan, np.nan
    pk = abs(int(np.nanargmax(np.where(ok, a, -1))) - int(np.nanargmax(np.where(ok, b, -1)))); pk = min(pk, 24 - pk)
    return float(np.corrcoef(a[ok], b[ok])[0, 1]), float(np.abs(a[ok] - b[ok]).mean()), pk


for p in T2_PANELS:
    P = load(p, "train"); mon = np.array([s[:7] for s in P.times]); MS = sorted(set(mon)); SH = {m: shape(P.Q[mon == m], [s for s in P.times if s.startswith(m)]) for m in MS}
    pool = lambda ex: np.nanmean(np.stack([SH[m] for m in MS if m != ex]), axis=0)
    for m in MS:
        pl = pool(m)
        for j in range(P.n):
            c, d, k = cmp(SH[m][:, j], pl[:, j]); rows.append(dict(panel=p, link=j, month=m, kind="train", corr=c, l1=d, peak=k))
    pl = pool(None); load.cache_clear()
    for sp, lab in (("validation", "2031-03"), ("private", "2031-04")):
        Pv = load(p, sp); S = shape(Pv.Q, Pv.times)
        for j in range(Pv.n):
            c, d, k = cmp(S[:, j], pl[:, j]); rows.append(dict(panel=p, link=j, month=lab, kind="test", corr=c, l1=d, peak=k))
        if sp == "validation":
            H = heads(Pv.V, Pv.vcut); rr, jj = np.nonzero(H); cnt = np.zeros(Pv.n); rare = np.zeros(Pv.n)
            for r, j in zip(rr, jj):
                key = (p, pd.Timestamp(Pv.times[r][:10]).dayofweek >= 5, (r % 288) // 24); f = RT.get(key); cnt[j] += 1; rare[j] += f is not None and f[j] <= 0.1
            for j in range(Pv.n):
                rare_rows.append(dict(panel=p, link=j, births=cnt[j], rare=rare[j]))
        load.cache_clear()
    print(p, "done", flush=True)
D = pd.DataFrame(rows); D.to_csv("C:/Users/user/tfb/work/t2/g341_shape.csv", index=False)
print("per month (links pooled): median corr / mean |shape diff| / share of links with the peak moved >= 2 h")
for m, d in D.groupby("month"):
    print(f"  {m} ({d.kind.iloc[0]}): corr median {d['corr'].median():.3f} (p10 {d['corr'].quantile(0.1):.3f}), l1 {d.l1.mean():.3f}, peak >= 2 h {(d.peak >= 2).mean():.3f}, links {d['corr'].notna().sum()}", flush=True)
tr = D[D.kind == "train"].groupby("month").l1.mean()
print(f"training months' l1: mean {tr.mean():.3f}, max {tr.max():.3f}; March {D[D.month == '2031-03'].l1.mean():.3f}, April {D[D.month == '2031-04'].l1.mean():.3f}", flush=True)
R = pd.DataFrame(rare_rows).merge(D[D.month == "2031-03"][["panel", "link", "l1", "corr"]], on=["panel", "link"]); R = R[R.births > 0]
rho, pv = spearmanr(R.rare / R.births, R.l1, nan_policy="omit"); rho2, pv2 = spearmanr(R.rare, R.l1, nan_policy="omit")
print(f"March links with births {len(R)}: Spearman(rare share of births, shape l1) {rho:+.3f} (p {pv:.3f}); Spearman(rare births, l1) {rho2:+.3f} (p {pv2:.3f}); rare births total {int(R.rare.sum())} of {int(R.births.sum())}", flush=True)
