"""G76 (10/23, strategy §16 (d) premise): how often does an onset happen at a site that is rare in the train months, per
month? Onset events in the released (masked) data of the validation (March) and private (April) months: a row t where
some link is queued (observed, v <= vcut) and no observed cell of the corridor was queued in rows t-17 .. t-1 (the M3
onset shape: free T-12 .. T+5, queue at T+6 = t); events closer than 2 h on one panel are merged. Rarity f as G54 (train
months, panel x weekend x 2-hour slot of the origin t-6); an event is 'new-site' if every queued link at t has f <= 0.1.
Prints per month: events, new-site share, by weekday/weekend; the train months (Vm, the Task 1 mask simulation) for
reference with in-sample f (so its share is a lower bound).
usage: python g76_rare_rate.py"""
import sys, os
os.environ["TFB_T2CACHE"] = "2_noT_af"
sys.path.insert(0, r"C:/Users/user/tfb/src")
sys.argv = [sys.argv[0], "0"]
import numpy as np, pandas as pd
from cachelib import load, T2_PANELS
from t1_features import inputs
from t2_eval_oof import T

SELON = [k for k in range(T.NWIN) if T.META["is_sel"][k] and str(T.META["cond"][k]) == "queue_onset"]
tkey = lambda p, we, t: (str(p), bool(we), (int(t) % 288) // 24)
CNT, DEN = {}, {}
for k in SELON:
    lab, el = T.LAB[k]; key = tkey(T.META["panel"][k], pd.Timestamp(str(T.META["date"][k])[:10]).dayofweek >= 5, T.META["t"][k])
    CNT[key] = CNT.get(key, 0) + (lab[5] & el[5]).astype(float); DEN[key] = DEN.get(key, 0) + 1
rows = []
for split in ("validation", "private", "train"):
    for p in T2_PANELS:
        P = load(p, split)
        V = inputs(P)[0] if split == "train" else P.V
        q = np.isfinite(V) & (V <= P.vcut[None, :]); anyq = q.any(axis=1)
        last = -10 ** 9
        for t in range(17, P.T):
            if anyq[t] and not anyq[t - 17:t].any() and t - last >= 24:
                last = t; o = t - 6; we = pd.Timestamp(P.times[o][:10]).dayofweek >= 5; key = tkey(p, we, o)
                f = CNT[key] / DEN[key] if key in DEN else np.zeros(P.n)
                links = np.flatnonzero(q[t])
                rows.append(dict(split=split, panel=p, month=P.times[t][:7], we=we, new=bool((f[links] <= 0.1).all()), n=len(links)))
R = pd.DataFrame(rows); R.to_csv("C:/Users/user/tfb/work/t2/g76_rare_rate.csv", index=False)
for (sp, mo), g in R.groupby(["split", "month"]):
    print(f"{sp:10s} {mo}: events {len(g):4d}, new-site share {g.new.mean():.2f} (weekday {g[~g.we].new.mean():.2f}, weekend {g[g.we].new.mean():.2f})")
for sp, g in R.groupby("split"):
    print(f"{sp}: events {len(g)}, new-site share {g.new.mean():.3f}")
