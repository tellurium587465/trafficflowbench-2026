"""Review: are the holes inside queued regions (array order = milepost order) noise flips or structural links?"""
import sys, numpy as np
sys.path.insert(0, r"C:/Users/user/tfb/src")
from cachelib import load
from t2_windows import queue_matrix

for pnl in sys.argv[1:]:
    P = load(pnl, "train"); Qm = queue_matrix(P)
    ok = np.isfinite(P.V) & P.E
    r = P.V / P.vcut[None, :]
    L = np.zeros_like(Qm); R = np.zeros_like(Qm)
    L[:, 1:] = Qm[:, :-1]; R[:, :-1] = Qm[:, 1:]
    gap = ok & ~Qm & L & R                                   # non-queued eligible link between two queued links
    inside = ok & Qm & L & R
    rg = r[gap]
    print(f"{pnl}: n={P.n} queued cells {Qm.sum()} gap cells {gap.sum()} ({gap.sum()/max(Qm.sum(),1):.3f} of queued)")
    print("  gap v/vcut quantiles 10/25/50/75/90:", np.round(np.percentile(rg, [10, 25, 50, 75, 90]), 3),
          f" share within 1.016 (2 sigma of 0.8% noise): {np.mean(rg < 1.016):.3f}")
    gl = gap.sum(0); top = np.sort(gl)[::-1]
    print(f"  gap concentration: top-5 links hold {top[:5].sum()/max(gl.sum(),1):.3f}, top-10 {top[:10].sum()/max(gl.sum(),1):.3f}")
    # per-link: how often a link is a hole when both neighbours are queued
    both = ok & L & R
    rate = gap.sum(0) / np.maximum(both.sum(0), 1)
    j = np.argsort(-gl)[:6]
    for jj in j:
        nbl = P.lanes[max(jj - 1, 0)], P.lanes[jj], P.lanes[min(jj + 1, P.n - 1)]
        print(f"   link {jj}: holes {gl[jj]} rate {rate[jj]:.2f} lanes(up,self,dn) {nbl} vcut {P.vcut[jj]:.1f} vf_fd {P.vf[jj]:.1f} "
              f"cap/lane {P.cap[jj]/P.lanes[jj]:.0f} len {P.length[jj]:.2f} med v/vcut@hole {np.median(r[gap[:, jj], jj]):.3f}")
    # speed ratio of queued interior cells for reference
    print("  interior queued v/vcut quantiles 50/90:", np.round(np.percentile(r[inside], [50, 90]), 3))
