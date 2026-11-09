"""E4 sanity check (pre-registered STATE.md 10/22): does adding March days 1-15 to the training months help or harm the
M3 windows of March 16-31? attnglob FINAL seeds 0,1 trained on (a) nn_data_afdem and (b) nn_data_afdemM15 (same code,
same GPU), CPU-inferred on nn_test_m3 (probs_m3__e4s_{a,b}_s{0,1}.pkl). Post-processing as the M3 gate (current
evidence rule, far gate, k-day limit, trap 4). Pass (no harm): production-weighted T2 (b) - (a) >= -0.003.
Reference only (not a decision input): (a) against the older q24f_attnglob seeds 0,1 (same config, earlier code / stack).
usage: python t2_m3_e4.py"""
import sys
sys.argv = [sys.argv[0], "0"]
import pickle, numpy as np, pandas as pd
import t2_m3_gate as G
from m3_stats import summary, fmt

CUT = "2031-03-16"


def mean_probs(fmt_):
    ps = [pickle.load(open(G.W + f"probs_m3__{fmt_.format(s)}.pkl", "rb")) for s in (0, 1)]
    return {k: sum(p[k] for p in ps) / len(ps) for k in ps[0]}


ARMS = {"a": mean_probs("e4s_a_s{}"), "b": mean_probs("e4s_b_s{}"), "old": mean_probs("q24f_attnglob_s{}")}
rows = []
for w in G.WID:
    m = G.META[w]; P, _ = G.panel(m["panel"]); late = str(P.times[m["t"]])[:10] >= CUT
    assert late == (m["day"] >= 15), (w, m["day"])                    # day 0 = March 1
    if not late:
        continue
    pred = (lambda p_: G.onset_pred(p_, m, "cur")) if m["cond"] == "queue_onset" else (lambda p_: G.ongoing_pred(p_, m, True))
    rows.append(dict(cond=m["cond"], panel=m["panel"], day=m["day"], band="0.5-0.6" if m["cov"] < 0.6 else "0.6-0.7",
                     **{k: G.iou(pred(v[w]), m) for k, v in ARMS.items()}))
D = pd.DataFrame(rows); D.to_csv(G.W + "m3_e4_sanity.csv", index=False)
print("M3 windows dated", CUT, "or later:", D.cond.value_counts().to_dict())
print("means", {k: round(float(D[k].mean()), 4) for k in ARMS})
s = summary(D, "a", "b"); print(fmt("E4 sanity (b: + March 1-15) vs (a: train months)", s))
for band, g in D.groupby("band"):
    print(f"   band {band}: windows {len(g)}, b-a {(g.b - g.a).mean():+.4f}")
print(fmt("reference: (a) vs older q24f_attnglob s0,1", summary(D, "old", "a")))
print("verdict:", "no harm (pass)" if s["t2_weighted"] >= -0.003 else "harm (stop E4 production, back to the strategy session)")
