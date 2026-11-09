"""M1: does the train-month demand profile (t2_demand.py, weekday x time-of-day) still describe the test months?
Backtest inside each window's own history (inputs only, no labels): free-flowing links (v/vf >= 0.85 on all history rows),
level from rows T-12..T-10, forecast of row T-1 (45 min ahead):
   profile: level * P(dow, slot T-1) / P(dow, slot T-12..T-10)       persistence: mean flow of rows T-12..T-10
Train windows use the leave-one-month-out profile (as in the features); test windows the all-train-month profile.
Also the realised 45-min flow change vs the profile's change (correlation, slope) = the signal dem_shape30 carries.
"""
import sys, numpy as np, pandas as pd
sys.path.insert(0, r"C:/Users/user/tfb/src")
from cachelib import load, T2_PANELS
from common import REL
from t2_demand import profile, _dow
import os
os.environ["TFB_T2CACHE"] = "2_noT_af"
import t2_eval_oof as EV

M = EV.T.META; rng = np.random.default_rng(1)
out = []


def bt(P, pr, t, day):
    w = pr[_dow(day)]; s = t % 288; rows = [t - 12, t - 11, t - 10]
    V = P.V[t - 12:t] / P.vf[None, :]; Q = P.Q
    ok = np.all(np.isfinite(V) & (V >= 0.85), axis=0) & np.isfinite(Q[t - 1]) & np.all(np.isfinite(Q[rows]), axis=0)
    base = np.mean([w[(s - (t - r)) % 288] for r in rows], axis=0); lv = np.mean([Q[r] for r in rows], axis=0)
    ok &= np.isfinite(base) & (base > 200) & np.isfinite(w[(s - 1) % 288]) & (lv > 200)
    if ok.sum() < 3:
        return None
    fp = lv / base * w[(s - 1) % 288]; act = Q[t - 1]
    e_prof = np.abs(fp - act) / act; e_pers = np.abs(lv - act) / act
    ch_act = np.log(act / lv); ch_prof = np.log(w[(s - 1) % 288] / base)
    return dict(n=int(ok.sum()), e_prof=float(np.median(e_prof[ok])), e_pers=float(np.median(e_pers[ok])),
                ch_act=ch_act[ok], ch_prof=ch_prof[ok])


for p in T2_PANELS:
    PT = load(p, "train")
    ks = np.flatnonzero((M["panel"] == p) & M["is_sel"])
    for k in rng.choice(ks, min(150, len(ks)), replace=False):
        d = str(M["date"][k]); r = bt(PT, profile(PT, d[:7]), int(M["t"][k]), d)
        if r:
            out.append(dict(split="train", panel=p, **r))
    prall = profile(PT, None)
    for split in ("validation", "private"):
        P = load(p, split); ti = {s: i for i, s in enumerate(P.times)}
        for w in pd.read_csv(REL / "task2" / p / split / "window_index.csv").itertuples():
            r = bt(P, prall, ti[w.forecast_origin], w.date)
            if r:
                out.append(dict(split=split, panel=p, **r))
    load.cache_clear(); print(p, flush=True)
R = pd.DataFrame(out)
for sp, g in R.groupby("split"):
    a = np.concatenate(list(g.ch_act)); b = np.concatenate(list(g.ch_prof))
    slope = np.polyfit(b, a, 1)[0]
    print(f"{sp:10s} windows {len(g):4d}  median err profile {g.e_prof.median():.4f}  persistence {g.e_pers.median():.4f}  "
          f"ratio {g.e_prof.median() / g.e_pers.median():.3f}  corr(change, profile change) {np.corrcoef(a, b)[0, 1]:.3f}  slope {slope:.3f}")
print(R.groupby(["panel", "split"]).apply(lambda g: round(g.e_prof.median() / g.e_pers.median(), 3), include_groups=False).unstack().to_string())
