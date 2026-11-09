"""How does a real onset look row by row? For the first morning queue row r of each day (truth layer), the queued links
at r: speed / vcut at rows r-3..r+2 (median over links and days), plus the distance between the two runs when there are
two. Same for the twin replay (lmin 0.12, gen2 capacities). Abrupt (r-1 ~1.5x -> r <1x on several links at once) =
coarse generator time step / instantaneous recording; gradual = 5-min averages of a fine-step CTM.
usage: python onset_shape.py [panels]"""
import sys, numpy as np, pandas as pd, warnings
warnings.simplefilter("ignore")
sys.argv += [] if len(sys.argv) > 1 else ["D12_I5_S,D7_I10_E"]
_a = list(sys.argv); sys.argv = ["x", "NONE"]
exec(open(r"C:/Users/user/tfb/twin/cellsize_fidelity.py", encoding="utf-8").read().split("pans = ")[0])
sys.argv = _a
for p in sys.argv[1].split(","):
    P = load(p, "train"); D = P.T // 288
    months = sorted(set(str(x)[:7] for x in P.day[:D])); keep = [d for d in range(D) if str(P.day[d])[:7] in months[:6]]
    jr = np.isin(np.arange(P.T) // 288, keep); dists = gen2.site_dists(P, keep); cm = gen2.base_cap_mult(P, keep, dists)
    for name, V in (("real", P.V[:D * 288]), ("twin", replay(P, 0.12, cm, dists, jr))):
        R = V / P.vcut[None, :]; truth = (np.nan_to_num(V, nan=1e9) <= P.vcut[None, :]) & P.E[:D * 288]
        prof, gaps, upstream_prev = [], [], []
        for d in range(D):
            for s in range(72, 144):
                r = d * 288 + s
                if truth[r].any():
                    idx = np.flatnonzero(truth[r]); prof.append(np.nanmedian(R[r - 3:r + 3][:, idx], axis=1))
                    runs = np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1)
                    if len(runs) >= 2:
                        gaps.append(runs[1][0] - runs[0][-1] - 1)
                    break
        prof = np.array(prof)
        print(f"{p} {name}: v/vcut at rows r-3..r+2 (median over mornings) {np.round(np.nanmedian(prof, axis=0), 2).tolist()}; "
              f"two-run mornings {len(gaps)}/{len(prof)}, gap links median {np.median(gaps) if gaps else np.nan}", flush=True)
    load.cache_clear()
