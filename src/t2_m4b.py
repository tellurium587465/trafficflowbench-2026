"""M4b: tail proxy on the OFFICIAL March windows (PLAN_20261022 §2.4 item 2). MARCH ONLY - never for April windows
(749601: it reads rows after T of the scored windows; a diagnostic companion of the public LB, not a selector for April).
For each official validation window, the rows T+19..T+30 (just after the release blank) are observed again; Q_tail =
links observed queued in any of them, among the links observed there. The score of a submission's window = IoU between
its predicted set at T+30 (restricted to those links) and Q_tail (1 when both are empty). Pairs of submissions that
changed only Task 2 are compared with their public LB differences: sign agreement and correlation.
usage: python t2_m4b.py"""
import glob, os, numpy as np, pandas as pd
from cachelib import load, T2_PANELS
from common import REL
from t2_windows import queue_matrix

W = "C:/Users/user/tfb/work/"
LB = pd.read_csv(W + "lb_submissions_20261022.csv"); LB["tag"] = LB.fileName.str.extract(r"^(s\d+)_")[0]
LBS = LB.dropna(subset=["publicScore"]).drop_duplicates("tag").set_index("tag").publicScore.to_dict()
PAIRS = [("s40", "s42"), ("s42", "s45"), ("s45", "s46"), ("s46", "s49"), ("s46", "s47"), ("s46", "s52"), ("s52", "s60"),
         ("s60", "s61"), ("s60", "s63"), ("s60", "s64"), ("s60", "s65"), ("s60", "s67"), ("s67", "s69"), ("s69", "s70"),
         ("s70", "s72"), ("s63", "s66"), ("s46", "s51"), ("s46", "s56")]


def queue_file(tag):
    fs = [f for f in glob.glob(W + f"t2/queue_{tag}_*.parquet") if "raw" not in os.path.basename(f) and not os.path.basename(f).startswith(f"queue_{tag}on")]
    fs = [f for f in fs if os.path.basename(f).split("_")[1] == tag]
    return sorted(fs, key=len)[0] if fs else None


TAIL = {}
for p in T2_PANELS:
    P = load(p, "validation"); ti = {s: i for i, s in enumerate(P.times)}; Qm = queue_matrix(P); obs = np.isfinite(P.V) & P.E
    for r in pd.read_csv(REL / "task2" / p / "validation" / "window_index.csv").itertuples():
        t = ti[r.forecast_origin]; o = obs[t + 19:t + 31].any(axis=0); q = (Qm[t + 19:t + 31] & obs[t + 19:t + 31]).any(axis=0)
        last = pd.Timestamp(r.forecast_end).strftime("%Y-%m-%dT%H:%M:%SZ")
        TAIL[r.window_id] = (o, q, r.condition, last, list(map(str, P.links)))
    load.cache_clear()
print("official March windows", len(TAIL), "| mean share of links observed in the tail", round(float(np.mean([v[0].mean() for v in TAIL.values()])), 3))

SC = {}
def scores(tag):
    if tag in SC:
        return SC[tag]
    f = queue_file(tag)
    if f is None:
        return None
    q = pd.read_parquet(f); q = q[q.window_id.astype(str).str.contains("_validation_")]
    q["timestamp"] = pd.to_datetime(q.timestamp, utc=True).dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    out = {}
    for w, g in q.groupby("window_id"):
        o, qt, cond, last, links = TAIL[str(w)]
        s = g[g.timestamp == last].set_index("link_id").queue_pred.reindex(links).fillna(0).to_numpy() == 1
        u = ((s | qt) & o).sum(); out[str(w)] = (cond, 1.0 if u == 0 else float((s & qt & o).sum() / u))
    SC[tag] = out
    return out


rows = []
for a, b in PAIRS:
    sa, sb = scores(a), scores(b)
    if sa is None or sb is None or a not in LBS or b not in LBS:
        print("skip", a, b, "(no queue file or LB score)"); continue
    d = {c: np.mean([sb[w][1] - sa[w][1] for w in sa if sa[w][0] == c]) for c in ("queue_onset", "queue_ongoing")}
    t2 = 0.5 * (d["queue_onset"] + d["queue_ongoing"]); lb = LBS[b] - LBS[a]
    rows.append(dict(pair=f"{a}->{b}", lb=round(lb, 5), m4b_t2=round(t2, 4), onset=round(d["queue_onset"], 4), ongoing=round(d["queue_ongoing"], 4),
                     agree=np.sign(t2) == np.sign(lb) if abs(t2) > 1e-12 else None))
R = pd.DataFrame(rows); print(R.to_string(index=False))
small = R[R.lb.abs() < 0.05]
print(f"sign agreement (pairs with a nonzero M4b difference, excluding the empty-set decompositions): "
      f"{int(small.agree.fillna(False).sum())}/{int(small.agree.notna().sum())}; correlation M4b vs LB {small.m4b_t2.corr(small.lb):.2f}")
