"""Audit A8: quality columns. Joint distribution of pct_observed / is_observed / is_imputed / is_missing / eligibility
vs finite values, and whether the measurement noise depends on pct_observed (=> the target cell's released
pct_observed would be a usable Task 1 feature, and ineligible-but-finite cells usable Task 2 inputs)."""
import sys, glob, numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
REL = "C:/Users/user/tfb/data/kaggle_public/corridors/"
p = sys.argv[1] if len(sys.argv) > 1 else "D12_I5_N"
fs = sorted(glob.glob(REL + p + "/train/mainline_states/**/*.parquet", recursive=True))[200:230]
d = pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True)
d["fin"] = d.speed_kmh.notna()
d["pbin"] = pd.cut(d.pct_observed, [-1, 0, 49, 74, 99, 100])
print(p, "train unmasked, 30 days:", len(d), "rows")
print(pd.crosstab([d.pbin, d.is_observed, d.is_missing], [d.is_score_eligible, d.fin]))
# noise vs pct: second difference in time on the same link, free-flow only
d = d.sort_values(["link_id", "timestamp"])
for c in ("speed_kmh", "flow_vph"):
    x = d.groupby("link_id")[c]
    d[c + "_r"] = (d[c] - 0.5 * (x.shift(1) + x.shift(-1))) / d[c]
ff = d.speed_kmh > 100
tab = d[ff].groupby("pbin")[["speed_kmh_r", "flow_vph_r"]].agg(lambda s: np.nanstd(s) * 100)
tab["n"] = d[ff].groupby("pbin").size()
print("relative 2nd-difference sd (%) of free-flow cells by pct_observed bin:\n", tab.round(3))
