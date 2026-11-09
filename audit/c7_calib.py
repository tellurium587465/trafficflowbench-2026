"""C7: calibration of the final Task 1 speeds on the simulated months: mean truth vs mean prediction by predicted
v/vf bin (all targets), and a cross-panel isotonic-style recalibration (bin means fitted on half the panels, applied to
the other half) scored with the LB-like aggregation."""
import sys, glob, os, types, numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, r"C:/Users/user/tfb/src"); sys.path.insert(0, r"C:/Users/user/tfb/audit")
from cachelib import FAMILY
from c3_post import P_of, score

SIM = "C:/Users/user/tfb/audit/sim/"
BINS = np.array([0, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 2.0])
Zs = {os.path.basename(f)[:-4]: dict(np.load(f)) for f in sorted(glob.glob(SIM + "*.npz"))}
rows = []
for p, Z in Zs.items():
    m = Z["target"] & Z["E"] & np.isfinite(Z["TV"])
    r = (Z["V"] / Z["vf"][None, :])[m]; t = (Z["TV"] / Z["vf"][None, :])[m]
    b = np.digitize(r, BINS) - 1
    for k in range(len(BINS) - 1):
        s = b == k
        if s.any():
            rows.append(dict(panel=p, bin=k, n=int(s.sum()), pred=r[s].sum(), true=t[s].sum(), se=((r[s] - t[s]) ** 2).sum()))
d = pd.DataFrame(rows)
g = d.groupby("bin")[["n", "pred", "true", "se"]].sum()
g["lo"] = BINS[g.index]; g["mean_pred"] = g.pred / g.n; g["mean_true"] = g.true / g.n; g["bias_kmh_at_110"] = (g.mean_true - g.mean_pred) * 110
g["se_share"] = g.se / g.se.sum()
print(g[["lo", "n", "mean_pred", "mean_true", "bias_kmh_at_110", "se_share"]].round(4).to_string())
# cross-panel recalibration: shift each bin by the bias measured on the other half
pans = list(Zs); halves = [pans[0::2], pans[1::2]]; res = []
for h in (0, 1):
    fit = d[d.panel.isin(halves[1 - h])].groupby("bin")[["n", "pred", "true"]].sum(); shift = ((fit.true - fit.pred) / fit.n).to_dict()
    for p in halves[h]:
        Z = Zs[p]; P = P_of(Z)
        r = Z["V"] / Z["vf"][None, :]; b = np.clip(np.digitize(r, BINS) - 1, 0, len(BINS) - 2)
        sh = np.vectorize(lambda k: shift.get(k, 0.0))(b)
        V2 = np.where(Z["target"], (r + sh) * Z["vf"][None, :], Z["V"]); V2 = np.maximum(V2, Z["Q"] / Z["kj"][None, :])
        s0 = score(Z, P, Z["V"], Z["Q"]); s1 = score(Z, P, V2, Z["Q"])
        res.append(dict(panel=p, dS=s1[0] - s0[0], dP=s1[1] - s0[1]))
r = pd.DataFrame(res).set_index("panel"); fams = sorted({FAMILY[p] for p in pans})
dS = np.mean([np.mean([r.dS[p] for p in pans if FAMILY[p] == f]) for f in fams]); dP = r.dP.mean()
print(f"cross-panel speed recalibration: dS_state {dS:+.5f} dS_phys {dP:+.5f} dTotal {0.35*dS+0.15*dP:+.5f}; panels better {(0.35*r.dS+0.15*r.dP>0).sum()}/{len(r)}")
