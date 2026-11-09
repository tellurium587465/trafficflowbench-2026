"""G154 (11/01, ledger #117, strategy T2_DEEP_REVIEW): does the free-flow speed of the redrawn months match links.csv? The label threshold is 0.6 x links.csv
free speed, so a month whose free flow sits elsewhere would move the labels of that month. Per link: median of V on free cells / links free speed
(P.vcut / 0.6). Free cell = observed, eligible, in a row with no observed queue on the panel within +-6 rows. Panel value = the median over links.
Training months are the control: their own month-to-month scatter tells what 1 % means on the D7 links with slow free flow.
usage: py -B -X utf8 g154_vf_month.py [panel]"""
import sys
sys.dont_write_bytecode = True; sys.path.insert(0, r"C:/Users/user/tfb/src")
import numpy as np, pandas as pd
from cachelib import load, T2_PANELS

rows = []
for p in (sys.argv[1:] or T2_PANELS):
    for split in ("train", "validation", "private"):
        P = load(p, split); V = np.where(P.E, P.V, np.nan); vl = P.vcut / 0.6; q = (V <= P.vcut[None, :]).any(axis=1)
        nq = np.convolve(q.astype(int), np.ones(13, int), "same") == 0; mon = np.array([str(t)[:7] for t in P.times])
        for mo in np.unique(mon):
            with np.errstate(all="ignore"):
                r = np.nanmedian(V[(mon == mo) & nq], axis=0) / vl
            rows.append(pd.DataFrame(dict(panel=p, month=mo, split=split, link=np.arange(P.n), r=r)))
    load.cache_clear()
D = pd.concat(rows, ignore_index=True); D.to_csv(r"C:/Users/user/tfb/work/t2/g154_vf_month.csv", index=False)
print("panel medians of the link ratios"); print(D.groupby(["panel", "month"]).r.median().unstack("month").round(4).to_string())
tr = D[D.split == "train"].groupby(["panel", "link"]).r.median().rename("rt").reset_index(); X = D.merge(tr, on=["panel", "link"]); X["dev"] = (X.r / X.rt - 1).abs()
print("share of links with |r / r(train median) - 1| > 1 % per month"); print(X.assign(o=X.dev > 0.01).groupby(["month", "split"]).o.mean().round(4).to_string())
