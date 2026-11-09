"""Audit A7b: corridor-level month levels (is the test scenario globally different?): congestion rate, flow levels,
discharge flow inside queues (q/cap where v <= v_cut), share of links ever congested, and first-queue-of-day site share
at the train-dominant sites."""
import sys, numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, r"C:/Users/user/tfb/src")
from cachelib import load, T2_PANELS

rows = []
for p in T2_PANELS:
    for sp in ("train", "validation", "private"):
        P = load(p, sp)
        V = P.V.copy(); Q = P.Q.copy()
        if sp == "train":
            V[P.blank] = np.nan; Q[P.blank] = np.nan
        ok = P.E & np.isfinite(V)
        months = np.array([s[:7] for s in P.times])
        for m in np.unique(months):
            s = months == m
            v, q, o = V[s], Q[s], ok[s]
            cong = (v <= P.vcut[None, :]) & o
            qc = q / P.cap[None, :]
            D = s.sum() // 288
            first = []
            for d in range(D):
                c = cong[d * 288:(d + 1) * 288]; r = np.flatnonzero(c.any(axis=1))
                if len(r):
                    first.append(int(np.flatnonzero(c[r[0]])[0]))
            rows.append(dict(panel=p, split=sp, month=m, cong_rate=cong.sum() / o.sum(), q_disch=np.nanmedian(qc[cong]),
                             q_ff_peak=np.nanpercentile(np.where(o & ~cong, qc, np.nan), 95), links_cong=(cong.sum(0) > 20).mean(),
                             days_with_queue=len(first) / D, first_sites=first))
        load.cache_clear()
d = pd.DataFrame(rows)
# share of first-queue-of-day events at the panel's top-3 train sites (+-2 links)
out = []
for p, g in d.groupby("panel"):
    tr = np.concatenate([np.array(x) for x in g[g.split == "train"].first_sites])
    vals, cnt = np.unique(tr, return_counts=True); top = vals[np.argsort(-cnt)[:3]]
    for r in g.itertuples():
        fs = np.array(r.first_sites)
        out.append(min(1.0, np.mean([np.min(np.abs(top - f)) <= 2 for f in fs])) if len(fs) else np.nan)
d["first_at_train_top3"] = out
pd.set_option("display.width", 220)
print(d.drop(columns="first_sites").groupby(["panel", "split"]).mean(numeric_only=True).round(3).to_string())
print(d.drop(columns="first_sites").groupby("split").mean(numeric_only=True).round(3).to_string())
