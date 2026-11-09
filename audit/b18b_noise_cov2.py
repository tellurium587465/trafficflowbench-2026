"""B18b (10/29): B18 with the trend-robust estimator. For x = s + e (s smooth, e white):
Cov(dx_t, dx_{t+1}) = Cov(ds_t, ds_{t+1}) - Var(e) and Cov(dx_t, dx_{t+2}) = Cov(ds_t, ds_{t+2}) ~ Cov(ds_t, ds_{t+1})
-> Var(e) ~ Cov(dx_t, dx_{t+2}) - Cov(dx_t, dx_{t+1}). Same for cross terms (speed-flow, adjacent links). Free-flow rows
(t-1 .. t+2 observed, V >= 0.85 vf), training months, speed km/h and log flow. Pooled per panel (sums over links) for the
cross terms to avoid ratio blow-ups."""
import sys, numpy as np, pandas as pd
sys.path.insert(0, r"C:/Users/user/tfb/src")
from cachelib import load
from common import PANELS


def cov_lag(a, b, ok, L):
    x, y, m = a[:-L], b[L:], ok[:-L] & ok[L:]
    n = m.sum(0); mx = np.where(m, x, 0).sum(0) / np.maximum(n, 1); my = np.where(m, y, 0).sum(0) / np.maximum(n, 1)
    return np.where(m, (x - mx) * (y - my), 0).sum(0) / np.maximum(n - 1, 1), n


out = []
for p in PANELS:
    P = load(p, "train"); V, Q = P.V.astype(float), P.Q.astype(float)
    free = np.isfinite(V) & np.isfinite(Q) & (Q > 0) & (V >= 0.85 * P.vf[None, :]); lq = np.log(np.where(Q > 0, Q, np.nan))
    dv, dq = np.diff(V, axis=0), np.diff(lq, axis=0); ok = free[1:] & free[:-1]; ok3 = ok[:-1] & ok[1:]
    est = lambda a, b: cov_lag(a, b, ok, 2)[0] - cov_lag(a, b, ok, 1)[0]
    vv, qq = est(dv, dv), est(dq, dq); vq = (est(dv, dq) + est(dq, dv)) / 2
    n = cov_lag(dv, dv, ok, 2)[1]; k = n > 2000
    # adjacent links: shift columns
    okp = ok[:, :-1] & ok[:, 1:]
    def est_pair(a1, a2):
        c = lambda L: cov_lag(a1, a2, okp, L)[0]
        c2 = lambda L: cov_lag(a2, a1, okp, L)[0]
        return ((c(2) - c(1)) + (c2(2) - c2(1))) / 2
    av = est_pair(dv[:, :-1], dv[:, 1:]); aq = est_pair(dq[:, :-1], dq[:, 1:]); kp = k[:-1] & k[1:]
    out.append(dict(panel=p, sv=np.sqrt(np.median(np.clip(vv[k], 0, None))), sq=np.sqrt(np.median(np.clip(qq[k], 0, None))),
                    rho_vq=np.sum(vq[k]) / np.sqrt(np.sum(vv[k]) * np.sum(qq[k])),
                    rho_adj_v=np.sum(av[kp]) / np.sqrt(np.sum(vv[:-1][kp]) * np.sum(vv[1:][kp])),
                    rho_adj_q=np.sum(aq[kp]) / np.sqrt(np.sum(qq[:-1][kp]) * np.sum(qq[1:][kp])), links=int(k.sum())))
    load.cache_clear()
D = pd.DataFrame(out); print(D.round(4).to_string())
print(f"median over panels: speed {D.sv.median():.2f} km/h | flow {D.sq.median():.2%} | speed-flow noise corr {D.rho_vq.median():+.3f} | adjacent-link noise corr: speed {D.rho_adj_v.median():+.3f}, flow {D.rho_adj_q.median():+.3f}")
