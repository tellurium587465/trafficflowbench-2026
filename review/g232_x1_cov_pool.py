"""G232 (11/03, ledger #195 preparation): history coverage of every origin in the dense X1 pool (nn_data_march_sp1, built by build_march.py
with SP = 1): the share of observed eligible cells in rows T-12 .. T-1 (build_march's COV), and the spacing at which the origins with
coverage >= 0.7 number about as many as the current pool (2,519). Writes C:/tfb_data/tfb_nn/nn_data_march_sp1/cov.npy.
usage: py -B -X utf8 g232_x1_cov_pool.py [split=validation] [pool=nn_data_march_sp1]"""
import os, sys, numpy as np
os.environ["TFB_T2_NOT"] = "1"; os.environ["TFB_T2_ALLFIN"] = "1"
sys.path.insert(0, r"C:/Users/user/tfb/src")
from cachelib import load
from common import NN
SPLIT = sys.argv[1] if len(sys.argv) > 1 else "validation"; POOL = sys.argv[2] if len(sys.argv) > 2 else "nn_data_march_sp1"
M = {f: np.load(f"{NN}/{POOL}/{f}.npy", allow_pickle=True) for f in ("panel", "t", "cond")}; cov = np.zeros(len(M["t"]))
for p in np.unique(M["panel"]):
    P = load(p, SPLIT); obs = np.isfinite(P.V) & P.E; ocnt = np.concatenate([[0], np.cumsum(obs.sum(axis=1))]); ii = np.flatnonzero(M["panel"] == p)
    t = M["t"][ii].astype(int); cov[ii] = (ocnt[t] - ocnt[t - 12]) / (12 * P.n); load.cache_clear()
np.save(f"{NN}/{POOL}/cov.npy", cov)


def keep(mask, sp):
    out, last = [], {}
    for i in np.flatnonzero(mask):
        p, t, c = M["panel"][i], int(M["t"][i]), M["cond"][i]
        if t - last.get((p, c), -10 ** 9) >= sp:
            out.append(i); last[(p, c)] = t
    return np.array(out, int)


print(f"{POOL}: {len(cov)} origins, coverage quantiles 10/50/90 % {np.quantile(cov, [0.1, 0.5, 0.9]).round(3)}, >= 0.7: {int((cov >= 0.7).sum())}")
for sp in (1, 2, 3, 4, 6):
    k = keep(cov >= 0.7, sp); print(f"coverage >= 0.7 at spacing {sp}: {len(k)} windows (onset {int((M['cond'][k] == 'queue_onset').sum())})")
print(f"current pool (spacing 6, coverage >= 0.5): {len(keep(cov >= 0.5, 6))}")
