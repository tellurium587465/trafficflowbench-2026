"""Can the twin reproduce the real onset 'cliff' (v/vcut 1.65 -> 0.55 in one row, several links at once, two runs with a
one-link gap in ~70% of mornings)? CTM variants (ctm.Corridor.run options): mean vs snapshot recording, capacity drop
cdrop in {1.0, 0.76}. In memory, two panels, identification on months 0-5 (gen2 capacities), all mornings scored.
usage: python onset_shape2.py [panels]"""
import sys, numpy as np, pandas as pd, warnings
warnings.simplefilter("ignore")
sys.path.insert(0, r"C:/Users/user/tfb/src"); sys.path.insert(0, r"C:/Users/user/tfb/twin")
from cachelib import load
from ctm import Corridor, junction_ratios
import gen2


def replay(P, cm, dists, jr, snapshot, cdrop, seed=0):
    rng = np.random.default_rng(seed); C = Corridor(P, cap_mult=cm); D = P.T // 288
    RAT = junction_ratios(P, jr); dayt = np.array([pd.Timestamp(str(x)).dayofweek >= 5 for x in P.day[:D]]).astype(int)
    CR = C.cell_ratios(RAT[dayt])
    up = pd.Series(P.Q[:, 0].astype(float)).interpolate(limit_direction="both").to_numpy()[:D * 288].reshape(D, 288)
    capf = np.ones((D, 288, C.NC))
    for b, (mu, s) in dists.items():
        draw = np.exp(mu + s * rng.standard_normal(D))
        for c in np.unique(C.cell_of[b:b + 3]):
            capf[:, :, c] = np.minimum(capf[:, :, c], (draw / C.cap[c])[:, None])
    Vc, Fc = C.run(up, CR, capf, snapshot=snapshot, cdrop=cdrop)
    return Vc[:, :, C.cell_of].reshape(-1, P.n)


def shape(P, V, D):
    R = V / P.vcut[None, :]; truth = (np.nan_to_num(V, nan=1e9) <= P.vcut[None, :]) & P.E[:D * 288]
    prof, two, gaps, nl = [], 0, [], []
    for d in range(D):
        for s in range(72, 144):
            r = d * 288 + s
            if truth[r].any():
                idx = np.flatnonzero(truth[r]); prof.append(np.nanmedian(R[r - 3:r + 3][:, idx], axis=1)); nl.append(len(idx))
                runs = np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1)
                if len(runs) >= 2:
                    two += 1; gaps.append(runs[1][0] - runs[0][-1] - 1)
                break
    return np.round(np.nanmedian(np.array(prof), axis=0), 2).tolist(), len(prof), two / max(len(prof), 1), np.mean(nl), (np.median(gaps) if gaps else np.nan)


for p in (sys.argv[1] if len(sys.argv) > 1 else "D12_I5_S,D7_I10_E").split(","):
    P = load(p, "train"); D = P.T // 288
    months = sorted(set(str(x)[:7] for x in P.day[:D])); keep = [d for d in range(D) if str(P.day[d])[:7] in months[:6]]
    jr = np.isin(np.arange(P.T) // 288, keep); dists = gen2.site_dists(P, keep); cm = gen2.base_cap_mult(P, keep, dists)
    print(p, "real", shape(P, P.V[:D * 288], D), flush=True)
    for snap, cd in ((False, 1.0), (True, 1.0), (False, 0.76), (True, 0.76)):
        print(p, f"twin snapshot={snap} cdrop={cd}", shape(P, replay(P, cm, dists, jr, snap, cd), D), flush=True)
    load.cache_clear()
