"""C5: structure of the free-flow error (the largest reducible block in c2b): per-link bias, per-link-day bias,
per-time (corridor-wide) bias, and the rest vs the white-noise floor. Speed in km/h, flow per lane."""
import sys, glob, os, numpy as np, pandas as pd
sys.path.insert(0, r"C:/Users/user/tfb/audit")
from c2_decomp import kinds

rows = []
for f in sorted(glob.glob("C:/Users/user/tfb/audit/sim/*.npz")):
    p = os.path.basename(f)[:-4]; Z = np.load(f); K = kinds(Z)
    m = Z["target"] & Z["E"] & np.isfinite(Z["TV"]) & (K == 4)
    r, c = np.nonzero(m)
    for nm, e, fl in (("v", (Z["V"] - Z["TV"])[m], (0.008 * Z["TV"][m]) ** 2), ("q", ((Z["Q"] - Z["TQ"]) / Z["lanes"][None, :])[m], (0.023 * Z["TQ"][m] / Z["lanes"][c]) ** 2)):
        d = pd.DataFrame({"e": e, "link": c, "day": r // 288, "t": r})
        se = (e ** 2).sum()
        lb = d.groupby("link").e.transform("mean"); ldb = d.groupby(["link", "day"]).e.transform("mean"); tb = d.groupby("t").e.transform("mean")
        rows.append(dict(panel=p, ch=nm, n=len(e), rmse=np.sqrt(se / len(e)), floor_rmse=np.sqrt(fl.sum() / len(e)),
                         link_bias=(lb ** 2).sum() / se, linkday_bias=((ldb ** 2).sum()) / se, time_bias=(tb ** 2).sum() / se, mean_e=e.mean()))
d = pd.DataFrame(rows)
pd.set_option("display.width", 200)
print(d.round(4).to_string(index=False))
