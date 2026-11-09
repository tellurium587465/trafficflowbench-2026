"""G19: is the official window selector 'first passage from the month start' (chronological, >= 6 h apart), and does
that explain the test ongoing windows' small, young queues (G18)? On each train panel-month: all qualifying origins
(t2_windows rules, persistence at the T row), then (a) chronological greedy from the month start, first 5 per
condition = 'first-passage'; (b) the local pool (random order greedy, as t2_windows.build). Window statistics from the
12 history rows (noisy observations, eligible, v <= v_cut): queued cells, queued links in the last row, origin hour,
day of month. Test windows: the same statistics from the released window_history (no labels)."""
import sys, numpy as np, pandas as pd
sys.path.insert(0, r"C:/Users/user/tfb/src")
from cachelib import load, T2_PANELS
from common import REL
from t2_windows import H, F, iou

rng = np.random.default_rng(0); rows = []
for p in T2_PANELS:
    P = load(p, "train")
    Qm = (np.nan_to_num(P.V, nan=1e9) <= P.vcut[None, :]) & P.E
    obs = np.isfinite(P.V) & P.E; any_q = Qm.any(1)
    cnt = np.concatenate([np.zeros((1, P.n), np.int32), np.cumsum(Qm, 0, dtype=np.int32)])
    ocnt = np.concatenate([[0], np.cumsum(obs.sum(1))]); qrow = np.concatenate([[0], np.cumsum(any_q)])
    cand = {"queue_onset": [], "queue_ongoing": []}
    for t in range(H, P.T - F):
        if P.times[t - H][:10] != P.times[t + F][:10]:
            continue
        if (ocnt[t] - ocnt[t - H]) / (H * P.n) < 0.7 or qrow[t + F + 1] - qrow[t + 1] == 0:
            continue
        hq = cnt[t] - cnt[t - H]
        if qrow[t + F] - qrow[t - H] == 0 and any_q[t + F]:
            cand["queue_onset"].append(t)
        elif hq.sum() >= 2 and hq.max() >= 2:
            fut = Qm[t + 1:t + F + 1]
            if iou(np.broadcast_to(Qm[t], fut.shape), fut, P.E[t + 1:t + F + 1]) <= 0.9:
                cand["queue_ongoing"].append(t)
    mon = np.array([s[:7] for s in P.times])
    def stat(t, kind, cond):
        h = Qm[t - H:t]
        return dict(panel=p, kind=kind, cond=cond, nq_hist=int(h.sum()), nq_last=int(h[-1].sum()), hour=int(P.times[t][11:13]),
                    dom=int(P.times[t][8:10]))
    for cond, ts in cand.items():
        ts = np.array(ts)
        for m in np.unique(mon):
            tm = ts[mon[ts] == m]; ch = []
            for t in tm:                                     # chronological first passage
                if not ch or t - ch[-1] >= 72:
                    ch.append(t)
                if len(ch) == 5:
                    break
            rows += [stat(t, "first-passage", cond) for t in ch]
        ch = []
        for t in rng.permutation(ts):                        # local pool (random order greedy)
            if all(abs(t - o) >= 72 for o in ch):
                ch.append(int(t))
        rows += [stat(t, "random pool", cond) for t in ch]
    for sp in ("validation", "private"):
        w = pd.read_csv(REL / "task2" / p / sp / "window_index.csv"); h = pd.read_parquet(REL / "task2" / p / sp / "window_history.parquet")
        vc = dict(zip(P.links, P.vcut)); h["q"] = (h.speed_kmh <= h.link_id.astype(str).map(vc)) & (h.is_score_eligible == 1)
        for r in w.itertuples():
            g = h[h.window_id == r.window_id].sort_values("timestamp"); ts_ = sorted(g.timestamp.unique())
            last = g[g.timestamp == ts_[-1]]
            rows.append(dict(panel=p, kind=f"test {sp}", cond=r.condition, nq_hist=int(g.q.sum()), nq_last=int(last.q.sum()),
                             hour=int(str(r.forecast_origin)[11:13]), dom=int(str(r.forecast_origin)[8:10])))
R = pd.DataFrame(rows)
pd.set_option("display.width", 200)
print(R.groupby(["cond", "kind"])[["nq_hist", "nq_last", "hour", "dom"]].agg(["median", "mean"]).round(1))
print(R.groupby(["cond", "kind"]).size())
R.to_csv("C:/Users/user/tfb/work/t2/g19_windows.csv", index=False)
