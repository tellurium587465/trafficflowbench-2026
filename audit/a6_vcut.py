"""Audit A6: which queue threshold reproduces the organizer's official TRAIN windows?
A = 0.6 * links.csv free_speed (what the team uses; = queue_utils fallback)   B = 0.6 * fd_parameters free_speed.
Official facts to reproduce: onset -> no queue in the visible history, first queue exactly at T+30;
ongoing -> visible history holds >=2 queued obs with >=2 on one link."""
import sys, numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, r"C:/Users/user/tfb/src")
from cachelib import load, T2_PANELS
from common import REL

res = []
for p in T2_PANELS:
    P = load(p, "train"); ti = {s: i for i, s in enumerate(P.times)}
    thr = {"A_links": P.vcut, "B_fd": 0.6 * P.vf}
    wi = pd.read_csv(REL / "task2" / p / "train" / "window_index.csv")
    for r in wi.itertuples():
        t = ti[r.forecast_origin]
        for nm, th in thr.items():
            Qm = (np.nan_to_num(P.V, nan=1e9) <= th[None, :]) & P.E
            h = Qm[t - 12:t]; fut = Qm[t + 1:t + 7]
            first = int(np.argmax(fut.any(axis=1))) + 1 if fut.any() else -1
            og_rule = h.sum() >= 2 and h.sum(axis=0).max() >= 2
            res.append(dict(panel=p, cond=r.condition, thr=nm, hist_q=int(h.sum()), og_rule=og_rule, first_step=first,
                            n_q_T30=int(fut[-1].sum()), mid_cells=int(((P.V[t + 1:t + 7] > P.vcut) & (P.V[t + 1:t + 7] <= 0.6 * P.vf) & P.E[t + 1:t + 7]).sum()),
                            q_cells=int((fut).sum())))
    load.cache_clear()
d = pd.DataFrame(res)
for nm, g in d.groupby("thr"):
    on = g[g.cond == "queue_onset"]; og = g[g.cond == "queue_ongoing"]
    print(f"{nm}: onset windows with NO history queue {int((on.hist_q == 0).sum())}/{len(on)}, first queue exactly at T+30 {int((on.first_step == 6).sum())}/{len(on)}, "
          f"onset that the rule would call ongoing {int(on.og_rule.sum())} | ongoing windows satisfying the rule {int(og.og_rule.sum())}/{len(og)}")
a = d[d.thr == "A_links"]
print("horizon cells with 0.6*vf_links < v <= 0.6*vf_fd (queued only under B):", int(a.mid_cells.sum()), "vs queued under A:", int(a.q_cells.sum()))
