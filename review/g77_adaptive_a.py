"""G77 (10/23, strategy §16 (d)): the causal month-adaptive mixing weight of the physics onset model vs the fixed a = 0.3.
M3 dev (exact): production onset blend (s46) + physics (probs_m3__physon) at a_w (physon_a_m3.pkl) vs a = 0.3 vs NN.
Train months (approximate, G70's per-window IoU on the grid a in {0, .1, .2, .3, .5}): a per window from that train
month's own completed earlier days (in-sample rarity -> the rate is a lower bound, so a is if anything too small), rounded
to the nearest grid value; vs the fixed 0.3 and vs NN.
usage: python g77_adaptive_a.py"""
import sys, os, json, pickle
sys.path.insert(0, r"C:/Users/user/tfb/src")
argv0 = sys.argv[0]
import numpy as np, pandas as pd
import t2_phys_onset as PO
sys.argv = [argv0, "0"]
import t2_m3_gate as G, t2_m3v2_gate as V, m3v2

W = G.W
spec = json.load(open(r"C:/Users/user/tfb/src/t2_specs.json"))["s37"]["spec"]
parts = [(x.split(":")[0], float(x.split(":")[1]) if ":" in x else 1.0) for x in spec.split("+")]
PR = {t: pickle.load(open(W + f"probs_m3__{t}.pkl", "rb")) for t, _ in parts}
PH = pickle.load(open(W + "probs_m3__physon.pkl", "rb")); AM = pickle.load(open(W + "physon_a_m3.pkl", "rb"))
WON = [w for w in G.WID if G.META[w]["cond"] == "queue_onset" and not m3v2.is_holdout(G.META[w]["day"])]
cols = {"nn": {}, "fixed": {}, "adaptive": {}}
for w in WON:
    m = G.META[w]; p = sum(wt * PR[t][w] for t, wt in parts) / sum(wt for _, wt in parts)
    cols["nn"][w] = V.score(p, m)
    for name, a in (("fixed", 0.3), ("adaptive", AM[w])):
        q = p.copy(); q[5] = (1 - a) * p[5] + a * PH[w][5]; cols[name][w] = V.score(q, m)
D = V.frame(cols).dropna(subset=list(cols))
for a_, b_ in (("nn", "fixed"), ("nn", "adaptive"), ("fixed", "adaptive")):
    print(m3v2.fmt(f"M3 dev onset {a_} -> {b_}", m3v2.summary(D, a_, b_, "private")))

# train months: adaptive a per window from the month's completed earlier days
os.environ["TFB_T2CACHE"] = "2_noT_af"
from t2_eval_oof import T
F_ = PO.rarity_table(); ev = PO.month_events("train", F_)
from cachelib import load
times = load("D7_I10_E", "train").times                                # same time axis for every panel (checked below)
mon_of_day = {}
for t in range(0, len(times), 288):
    mon_of_day[t // 288] = times[t][:7]
R = pd.read_csv(W + "g70_phys_lomo.csv"); GRID = np.array([0.0, 0.1, 0.2, 0.3, 0.5]); aa = []
for r in R.itertuples():
    t = int(T.META["t"][r.k]); d0 = t // 288; mo = mon_of_day[d0]
    e = [(d, nw) for d, nw in ev if mon_of_day.get(d) == mo]
    aa.append(PO.adaptive_a(e, d0))
R["a_adapt"] = aa; R["a_grid"] = GRID[np.abs(GRID[None, :] - R.a_adapt.values[:, None]).argmin(axis=1)]
R["adaptive"] = [R.loc[i, f"a{g}"] for i, g in zip(R.index, R.a_grid)]
print(f"train months: adaptive a median {R.a_adapt.median():.3f} (grid share {R.a_grid.value_counts(normalize=True).round(2).to_dict()})")
for name, g in (("all", R), ("usual", R[~R.new]), ("new-site", R[R.new])):
    print(f"  {name}: fixed 0.3 {(g['a0.3'] - g['a0.0']).mean():+.4f} | adaptive {(g['adaptive'] - g['a0.0']).mean():+.4f} (n {len(g)})")
