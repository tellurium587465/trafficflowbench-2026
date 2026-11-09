"""Are the bottleneck capacities structural or redrawn by the scenario? (10/14) Inside a congested block the real data
sit on the congested FD branch at the bottleneck's discharge flow (D7_I10_E block 4-10: q/cap 0.74-0.75, v/vf 0.31-0.34
for hours), so the discharge flow of each site is directly observable. Aggregate distribution check (no labels, visible
cells only): per block congested >= 30 cells in a split, median q/cap and v/vf over its congested cells, train months
(mainline_states) vs the validation month and the private month (mainline_states_masked, visible cells).
Same levels -> capacities are structural (the test month's onset site = flows against fixed capacities); different ->
redrawn. Blocks = block_structure.py on the train months (structural). Writes work/twin_capacity_shift.csv.
usage: python capacity_shift.py [panels=ALL T2 panels]"""
import sys, glob, numpy as np, pandas as pd, warnings
warnings.simplefilter("ignore")
sys.path.insert(0, r"C:/Users/user/tfb/src"); sys.path.insert(0, r"C:/Users/user/tfb/twin")
from cachelib import load, T2_PANELS
from common import REL
from onset_shape3_lib import raw_matrices, blocks, split_cells


out = []
for p in (sys.argv[1].split(",") if len(sys.argv) > 1 else T2_PANELS):
    P = load(p, "train"); V, Q, _ = raw_matrices(P); bid = blocks(V, P)
    ok = np.isfinite(V) & np.isfinite(Q); cg = ok & (V <= P.vcut[None, :])
    r, c = np.nonzero(cg); tr = pd.DataFrame(dict(b=bid[c], u=Q[r, c] / P.cap[c], l=V[r, c] / P.vf[c]))
    res = {"train": tr}
    for sp in ("validation", "private"):
        c2, v2, q2 = split_cells(P, sp); m = v2 <= P.vcut[c2]
        res[sp] = pd.DataFrame(dict(b=bid[c2[m]], u=q2[m] / P.cap[c2[m]], l=v2[m] / P.vf[c2[m]]))
    agg = {sp: d.groupby("b").agg(n=("u", "size"), u=("u", "median"), l=("l", "median")) for sp, d in res.items()}
    J = agg["train"].join(agg["validation"], rsuffix="_val").join(agg["private"], rsuffix="_prv")
    J = J[(J.n >= 30) & (J.n_val >= 30) & (J.n_prv >= 30)]
    for b, row in J.iterrows():
        out.append(dict(panel=p, block=int(b), **{k: round(float(v), 3) for k, v in row.items()}))
    if len(J):
        print(f"{p}: blocks congested in all three {len(J)} | median |du| val {np.median(np.abs(J.u_val - J.u)):.3f} prv {np.median(np.abs(J.u_prv - J.u)):.3f}"
              f" | median |dl| val {np.median(np.abs(J.l_val - J.l)):.3f} prv {np.median(np.abs(J.l_prv - J.l)):.3f}", flush=True)
        print("   " + J[["n", "u", "u_val", "u_prv", "l", "l_val", "l_prv"]].round(3).head(8).to_string().replace("\n", "\n   "), flush=True)
    load.cache_clear(); del V, Q
pd.DataFrame(out).to_csv(r"C:/Users/user/tfb/work/twin_capacity_shift.csv", index=False)
