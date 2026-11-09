"""Is occupancy released at Task 1 target cells, and how well does it pin down density there?

1) validation: share of target cells whose occupancy is finite (speed/flow are blanked).
2) train: density k = q/v vs occupancy. Per-link fit k ~ a_l * occ (fitted on free-flow cells) and its relative
   error on congested vs free-flow cells, compared with a crude neighbour-interpolation density estimate.
"""
import numpy as np
from cachelib import load, PANELS

for p in PANELS[:6]:
    Pv = load(p, "validation")
    tgt = Pv.target
    fin = np.isfinite(Pv.O[tgt]).mean(); finv = np.isfinite(Pv.V[tgt]).mean()
    P = load(p, "train"); ok = P.E & (P.P >= 75) & np.isfinite(P.V) & np.isfinite(P.Q) & np.isfinite(P.O) & (P.O > 0)
    k = np.where(ok, P.Q / np.maximum(P.V, 1), np.nan); o = np.where(ok, P.O, np.nan)
    free = ok & (P.V > 0.9 * P.vf[None, :]); cong = ok & (P.V < 0.6 * P.vf[None, :])
    a = np.nanmedian(np.where(free, k / o, np.nan), axis=0)              # per-link k/occ ratio from free flow
    kh = a[None, :] * o
    rel = lambda m: np.nanmedian(np.abs(kh[m] / k[m] - 1))
    # neighbour interpolation of density (same time, links i-1 and i+1)
    kn = np.full_like(k, np.nan); kn[:, 1:-1] = np.nanmean(np.stack([k[:, :-2], k[:, 2:]]), axis=0)
    reln = lambda m: np.nanmedian(np.abs(kn[m] / k[m] - 1))
    print(f"{p:11s} val target cells: occ finite {fin:.3f} (speed finite {finv:.3f}) | train occ->k rel.err free {rel(free):.3f} "
          f"cong {rel(cong):.3f} | neighbour-mean k rel.err free {reln(free):.3f} cong {reln(cong):.3f} | cong share {cong.sum()/ok.sum():.3f}", flush=True)
