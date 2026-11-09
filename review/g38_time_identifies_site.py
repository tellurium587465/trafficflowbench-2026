"""G38: does the time of the first morning queue identify WHICH site breaks (beyond the site frequency)? Each site's
breakdown time = when its demand curve crosses its capacity, a structural property that may survive a month change
better than the frequencies. G34 events; sites = head // 3 buckets; fit on months 0-5, test on months 6-8 (the real
shift between them is small, so this is the in-distribution value of time; the test months move frequencies more).
Predict the site by (a) the most frequent site, (b) argmax P(site) x N(slot; mean_s, sd_s), (c) argmax N(...) only.
usage: python g38_time_identifies_site.py"""
import sys, numpy as np, pandas as pd
from scipy.stats import norm
sys.path.insert(0, r"C:/Users/user/tfb/src")
from cachelib import load
E = pd.read_csv(r"C:/Users/user/tfb/work/t2/g34_events.csv"); E["site"] = E.b // 3
rows = []
for p, g in E.groupby("panel"):
    P = load(p, "train"); mon = np.array([str(P.day[d])[:7] for d in g.day]); g = g.assign(mon=mon)
    tr = g[g.mon < "2030-12"]; te = g[g.mon >= "2030-12"]
    if len(tr) < 20 or len(te) < 5:
        continue
    pri_all = tr.site.value_counts(normalize=True)                   # review 10/13: the prior by day type (weekday/weekend)
    st = tr.groupby("site").slot.agg(["mean", "std", "size"]); st["std"] = st["std"].fillna(12).clip(lower=3)
    for r in te.itertuples():
        g_ = tr[tr.we == r.we]; pri = g_.site.value_counts(normalize=True).reindex(pri_all.index).fillna(0) if len(g_) >= 10 else pri_all
        pri = pri.sort_values(ascending=False)
        sites = pri.index.to_numpy()
        lik = np.array([norm.pdf(r.slot, st.loc[s, "mean"], st.loc[s, "std"]) for s in sites])
        post = pri.to_numpy() * lik
        hit = lambda s: abs(int(s) - int(r.site)) <= 1
        rows.append(dict(panel=p, prior=hit(sites[0]), posterior=hit(sites[np.argmax(post)]), time_only=hit(sites[np.argmax(lik)]),
                         known=r.site in set(sites)))
R = pd.DataFrame(rows)
print(R.groupby("panel")[["prior", "posterior", "time_only", "known"]].mean().round(3).to_string())
print("all:", R[["prior", "posterior", "time_only", "known"]].mean().round(3).to_dict(), "n", len(R))
