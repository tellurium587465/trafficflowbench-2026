"""TTA on M3 (pre-registered STATE.md 10/22 10:48): the NOMEM blend (q24f, 5 configurations x seeds 0-3, weights
2,2,2,1,1) computed on each of the 8 perturbed feature sets (t2_m3_tta_build.py; probs_m3tta<k>__<tag>.pkl), averaged
in logit space over the draws (cells that are 0 in every draw stay 0), against the current NOMEM blend on the
unperturbed features. Post-processing as the M3 gate. Pass per condition: production-weighted difference >= +2 SE.
usage: python t2_m3_tta.py"""
import sys
sys.argv = [sys.argv[0], "0,1,2,3"]
import pickle, numpy as np, pandas as pd
import t2_m3_gate as G
from m3_stats import summary, fmt

K = 8
lg = lambda x: np.log(np.clip(x, 1e-6, 1 - 1e-6) / (1 - np.clip(x, 1e-6, 1 - 1e-6)))


def blend(prefix):
    parts = []
    for fam, w in G.CFG5("q24f"):
        for s in G.SEEDS:
            parts.append((w, pickle.load(open(G.W + f"probs_{prefix}__{fam}_s{s}.pkl", "rb"))))
    tw = sum(w for w, _ in parts)
    return {k: sum(w * pr[k] for w, pr in parts) / tw for k in parts[0][1]}


draws = [blend(f"m3tta{k}") for k in range(1, K + 1)]
TTA = {}
for w in G.WID:
    st = np.stack([d[w] for d in draws]); z = (st == 0).all(axis=0)
    TTA[w] = np.where(z, 0.0, 1 / (1 + np.exp(-lg(st).mean(axis=0))))
rows = []
for w in G.WID:
    m = G.META[w]
    pred = (lambda p_: G.onset_pred(p_, m, "cur")) if m["cond"] == "queue_onset" else (lambda p_: G.ongoing_pred(p_, m, True))
    rows.append(dict(cond=m["cond"], panel=m["panel"], day=m["day"], band="0.5-0.6" if m["cov"] < 0.6 else "0.6-0.7",
                     cur=G.iou(pred(G.NOMEM[w]), m), tta=G.iou(pred(TTA[w]), m)))
D = pd.DataFrame(rows); D.to_csv(G.W + "m3_tta.csv", index=False)
print("mean |p_tta - p_cur| over cells:", round(float(np.mean([np.abs(TTA[w] - G.NOMEM[w]).mean() for w in G.WID])), 5),
      "| windows whose IoU changed:", int(sum((pred_ != 0) for pred_ in (D.tta - D.cur))))
for cond in ("queue_onset", "queue_ongoing"):
    s = summary(D[D.cond == cond], "cur", "tta"); print(fmt(f"{cond} TTA vs current", s))
    print("   verdict:", "adopt candidate (>= +2 SE)" if s["t2_weighted"] >= 2 * s["se"] else "not adopted")
    for band, gb in D[D.cond == cond].groupby("band"):
        print(f"   band {band}: windows {len(gb)}, tta-current {(gb.tta - gb.cur).mean():+.4f}")
print(fmt("T2 TTA vs current", summary(D, "cur", "tta")))
