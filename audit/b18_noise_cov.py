"""B18 (10/29, strategy upper-bound audit item 1): the observation noise re-estimated from the lag-1 covariance of first
differences. For a smooth state plus white noise e_t, Cov(dx_t, dx_{t+1}) = -Var(e) exactly (the state's increments are
nearly uncorrelated with the noise and slow), so sigma^2 = -Cov(dx_t, dx_{t+1}). B8 used the spread around the link median
(speed; counts the slow real variation) and adjacent-link differences (flow; counts real link-to-link differences), which
both over-state the floor (the model sat below it on 96 % of the free-flow cells, pred/floor 0.90).
Training months, free-flow triples (rows t-1, t, t+1 observed, V >= 0.85 vf) per link: speed in km/h (additive) and flow
as log q (multiplicative). Also (a) the speed-flow noise correlation from the cross lag-1 covariances,
rho = -[Cov(dv_t, dlq_{t+1}) + Cov(dlq_t, dv_{t+1})] / 2 / (sigma_v sigma_lq), and (b) adjacent links (j, j+1): the
same-time correlation of the differences and the noise correlation from the cross lag-1 covariance.
Writes work/b18_noise.csv (per panel x link). usage: py b18_noise_cov.py"""
import sys, numpy as np, pandas as pd
sys.path.insert(0, r"C:/Users/user/tfb/src")
from cachelib import load
from common import PANELS, WORK


def lagcov(a, b, ok):
    """per column: Cov(a_t, b_{t+1}) over rows where ok_t and ok_{t+1} (a, b are first differences)"""
    x, y, m = a[:-1], b[1:], ok[:-1] & ok[1:]
    n = m.sum(0); mx = np.where(m, x, 0).sum(0) / np.maximum(n, 1); my = np.where(m, y, 0).sum(0) / np.maximum(n, 1)
    return np.where(m, (x - mx) * (y - my), 0).sum(0) / np.maximum(n - 1, 1), n


rows, adj = [], []
for p in PANELS:
    P = load(p, "train"); V, Q = P.V.astype(float), P.Q.astype(float)
    free = np.isfinite(V) & np.isfinite(Q) & (Q > 0) & (V >= 0.85 * P.vf[None, :])
    lq = np.log(np.where(Q > 0, Q, np.nan))
    dv, dq = np.diff(V, axis=0), np.diff(lq, axis=0); ok = free[1:] & free[:-1]          # difference t = x_{t+1} - x_t
    cvv, n = lagcov(dv, dv, ok); cqq, _ = lagcov(dq, dq, ok)
    cvq, _ = lagcov(dv, dq, ok); cqv, _ = lagcov(dq, dv, ok)
    sv = np.sqrt(np.clip(-cvv, 0, None)); sq = np.sqrt(np.clip(-cqq, 0, None))
    rho = -(cvq + cqv) / 2 / np.maximum(sv * sq, 1e-12)
    var_dv = np.array([np.var(dv[ok[:, j], j]) if ok[:, j].sum() > 2 else np.nan for j in range(P.n)])
    for j in range(P.n):
        if n[j] > 2000:
            rows.append(dict(panel=p, link=j, n=int(n[j]), sv_kmh=sv[j], sq_rel=sq[j], rho_vq=rho[j],
                             lag1_ac_dv=cvv[j] / var_dv[j] if var_dv[j] > 0 else np.nan, vf=P.vf[j]))
    for j in range(P.n - 1):                                     # (b) adjacent links
        m = ok[:, j] & ok[:, j + 1]
        if m.sum() < 2000:
            continue
        same = np.corrcoef(dv[m, j], dv[m, j + 1])[0, 1]
        mm = m[:-1] & m[1:]
        c1 = np.cov(dv[:-1, j][mm], dv[1:, j + 1][mm])[0, 1]; c2 = np.cov(dv[:-1, j + 1][mm], dv[1:, j][mm])[0, 1]
        noise_corr = -(c1 + c2) / 2 / max(sv[j] * sv[j + 1], 1e-12)
        adj.append(dict(panel=p, link=j, same_time_corr_dv=same, noise_corr_v=noise_corr))
    load.cache_clear()
D = pd.DataFrame(rows); D.to_csv(WORK + "/b18_noise.csv", index=False); A = pd.DataFrame(adj)
print(f"links {len(D)} | lag-1 autocorrelation of the speed differences: median {D.lag1_ac_dv.median():+.3f} (white noise on a smooth state: -0.5)")
g = D.groupby("panel").agg(sv=("sv_kmh", "median"), sq=("sq_rel", "median"), rho=("rho_vq", "median"), ac=("lag1_ac_dv", "median"))
print(g.round(4).to_string())
print(f"all links: speed noise median {D.sv_kmh.median():.2f} km/h (IQR {D.sv_kmh.quantile(.25):.2f}-{D.sv_kmh.quantile(.75):.2f}), "
      f"as a share of vf {np.median(D.sv_kmh / D.vf):.2%} | flow noise median {D.sq_rel.median():.2%} (IQR {D.sq_rel.quantile(.25):.2%}-{D.sq_rel.quantile(.75):.2%})")
print(f"(a) speed-flow noise correlation: median {D.rho_vq.median():+.3f} (IQR {D.rho_vq.quantile(.25):+.3f} to {D.rho_vq.quantile(.75):+.3f})")
print(f"(b) adjacent links: same-time correlation of the speed differences median {A.same_time_corr_dv.median():+.3f}; noise correlation from the cross lag-1 covariance median {A.noise_corr_v.median():+.3f}")
