import sys; sys.path.insert(0, r"C:/Users/user/tfb/src")
import numpy as np
from cachelib import load
for p in ("D12_I5_S", "D7_I10_E"):
    P = load(p, "train"); vr = P.V / P.vf[None, :]; ok = np.isfinite(vr) & P.E
    c = vr[ok & (P.V <= P.vcut[None, :])]
    h, e = np.histogram(c, bins=np.arange(0, 0.62, 0.02))
    print(p, "congested cells", c.size, "v/vf quantiles 5/25/50/75/95:", np.round(np.quantile(c, [.05, .25, .5, .75, .95]), 3).tolist())
    print("  hist (0.02 bins from 0):", (h / h.sum()).round(3).tolist())
    q = P.Q / P.cap[None, :]; qc = q[ok & (P.V <= P.vcut[None, :])]
    print("  congested flow / capacity quantiles 5/25/50/75/95:", np.round(np.nanquantile(qc, [.05, .25, .5, .75, .95]), 3).tolist())
    load.cache_clear()
