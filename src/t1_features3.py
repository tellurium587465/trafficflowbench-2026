"""Task 1 features v3 = v2 + density-space kernels and FD-implied speed/flow."""
import numpy as np
from t1_features2 import grid_features2, rows2, base_of2, inputs2, link_statics
from t1_features import kernel_mean, prev_next, take

def grid_features3(P):
    F, V, Q, ok = grid_features2(P)
    lanes = P.lanes[None, :].astype(np.float32)
    k = Q / np.maximum(V, 1.0) / lanes                       # veh/km/lane
    x = np.concatenate([[0.0], np.cumsum(P.length[:-1])])
    vfree = float(np.nanmedian(P.vf)); wcong = float(np.nanmedian(P.w))
    kc = (P.kc / P.lanes)[None, :]; kj = (P.kj / P.lanes)[None, :]; w = P.w[None, :]
    vfl = (P.vf * link_statics(P.panel, P.split)[0])[None, :]
    p, nx = prev_next(V, ok)
    kp, kn = take(k, p), take(k, nx)
    gp, gn = F["gap_prev"], F["gap_next"]
    wp = np.where(np.isfinite(kp), 1 / np.maximum(gp, 1), 0); wn = np.where(np.isfinite(kn), 1 / np.maximum(gn, 1), 0)
    with np.errstate(all="ignore"):
        F["lin_k"] = (np.nan_to_num(kp) * wp + np.nan_to_num(kn) * wn) / (wp + wn)
    F["cg_s05_k"], _ = kernel_mean(k, ok, x, 0.5, [-3, -2, -1, 0, 1, 2, 3], -wcong, 2.5 / 60)
    F["ff_s05_k"], _ = kernel_mean(k, ok, x, 0.5, [-2, -1, 0, 1, 2], vfree, 2.5 / 60)
    for nm in ("lin_k", "cg_s05_k"):
        kk = F[nm]
        cong = kk > kc
        with np.errstate(all="ignore"):
            qfd = np.where(cong, w * (kj - kk), vfl * kk)          # per lane
            vfd = np.where(cong, qfd / np.maximum(kk, 1e-3), vfl)
        F[f"{nm}_vr_fd"] = vfd / P.vf[None, :]; F[f"{nm}_ql_fd"] = qfd
    return F, V, Q, ok

rows3 = rows2
base_of3 = base_of2
