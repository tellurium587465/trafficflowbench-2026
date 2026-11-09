"""G79 = M4b (INSTRUMENT_V3_SPEC §2.4, 10/23): the tail proxy on the 40 OFFICIAL March onset windows - the only onset
yardstick with complete official histories. GREY: it reads March rows after T (the released rows after the window's blank,
T+19 .. T+30 = 95-150 min) to judge methods; March diagnostics only, NEVER for April (749601: selection by post-T data),
and only 40 windows. For each window: Q_tail = links queued (observed, v <= vcut) in >= 1 of the tail rows; the
submission's T+30 set S (queue parquet); (a) S hits Q_tail (any overlap of S with Q_tail, +-1 link), (b) precision
|S within +-1 of Q_tail| / |S|, (c) rarity of Q_tail (f from the train months, panel x weekend x 2-hour slot of the
origin; share of Q_tail links with f <= 0.1, and whether its downstream head link is rare) -> the share of new-site onsets
among the official March onset windows (for the raking target of §2.2, March version).
usage: python g79_m4b.py <queue name>[,<queue name>...]     e.g. s76,s77p"""
import sys, os
os.environ["TFB_T2CACHE"] = "2_noT_af"
sys.path.insert(0, r"C:/Users/user/tfb/src")
names = sys.argv[1].split(","); sys.argv = [sys.argv[0], "0"]
import numpy as np, pandas as pd
from cachelib import load, T2_PANELS
from common import REL
from t2_eval_oof import T

W = "C:/Users/user/tfb/work/t2/"
key = lambda p, we, t: (str(p), bool(we), (int(t) % 288) // 24)
CNT, DEN = {}, {}
for k in range(T.NWIN):
    if T.META["is_sel"][k] and str(T.META["cond"][k]) == "queue_onset":
        lab, el = T.LAB[k]; kk = key(T.META["panel"][k], pd.Timestamp(str(T.META["date"][k])[:10]).dayofweek >= 5, T.META["t"][k])
        CNT[kk] = CNT.get(kk, 0) + (lab[5] & el[5]).astype(float); DEN[kk] = DEN.get(kk, 0) + 1
QS = {n: pd.read_parquet(W + f"queue_{n}.parquet") for n in names}
rows = []
for p in T2_PANELS:
    P = load(p, "validation"); ti = {s: i for i, s in enumerate(P.times)}; li = {l: i for i, l in enumerate(P.links)}
    q = np.isfinite(P.V) & (P.V <= P.vcut[None, :])
    for x in pd.read_csv(REL / "task2" / p / "validation" / "window_index.csv").itertuples():
        if x.condition != "queue_onset":
            continue
        t = ti[x.forecast_origin]; tail = q[t + 19:t + 31].any(axis=0); obs = np.isfinite(P.V[t + 19:t + 31]).any(axis=0)
        we = pd.Timestamp(P.times[t][:10]).dayofweek >= 5; kk = key(p, we, t)
        f = CNT[kk] / DEN[kk] if kk in DEN else np.zeros(P.n)
        near = tail.copy(); near[1:] |= tail[:-1]; near[:-1] |= tail[1:]
        r = dict(w=x.window_id, panel=p, n_tail=int(tail.sum()), tail_obs=float(obs.mean()),
                 tail_rare=float((f[tail] <= 0.1).mean()) if tail.any() else np.nan,
                 head_rare=bool(f[np.flatnonzero(tail).max()] <= 0.1) if tail.any() else None)
        for n, Qd in QS.items():
            g = Qd[(Qd.window_id == x.window_id)]
            ts = sorted(g.timestamp.unique()); g = g[(g.timestamp == ts[-1]) & (g.queue_pred == 1)]   # the T+30 step
            S = np.zeros(P.n, bool); S[[li[l] for l in g.link_id]] = True
            r[f"{n}_size"] = int(S.sum()); r[f"{n}_hit"] = bool((S & near).any()) if tail.any() else None
            r[f"{n}_prec"] = float((S & near).sum() / S.sum()) if (tail.any() and S.any()) else np.nan
        rows.append(r)
R = pd.DataFrame(rows); R.to_csv(W + "g79_m4b.csv", index=False)
V = R[R.n_tail > 0]
print(f"official March onset windows {len(R)}; with a queue in the tail rows {len(V)} (tail rows observed {R.tail_obs.mean():.2f})")
print(f"new-site share (all tail links rare) {np.mean(V.tail_rare == 1):.2f}, head link rare {V.head_rare.mean():.2f}, mean rare share {V.tail_rare.mean():.2f}")
for n in names:
    print(f"{n}: set size {R[f'{n}_size'].mean():.2f}, hits the tail {V[f'{n}_hit'].mean():.3f}, precision {V[f'{n}_prec'].mean():.3f}"
          f" | new-site windows: hits {V[V.tail_rare == 1][f'{n}_hit'].mean():.3f} (n {int((V.tail_rare == 1).sum())})")
