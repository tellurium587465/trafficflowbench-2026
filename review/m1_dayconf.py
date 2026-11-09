"""M1: day-level map conformity before T (input-only; the same code on train windows and on the test-window inputs).

U[b, j] = daily frequency of a queue at link j in 30-min bin b (other months, same day type; test: all 9 train months).
O[b, j] = queued (observed cell, v <= vcut) at link j in bin b today, rows from 00:00 to T-1.
  R_day = sum(O*U) / sum(U * obs)   share of the usual queue map that came true today      (recall-like)
  P_day = sum(O*U) / sum(O)         how usual today's realized queue places were           (precision-like)
  Rfar  = R_day restricted to links > 3 from the visible queue (where far predictions live)
usage: python m1_dayconf.py -> m1_dayconf.pkl
"""
import os, sys
os.environ["TFB_T2CACHE"] = "2_noT_af"
sys.path.insert(0, r"C:/Users/user/tfb/src")
import numpy as np, pandas as pd
from cachelib import load, T2_PANELS
from common import REL
from t2_windows import queue_matrix

W = pd.read_pickle(r"C:/Users/user/tfb/review/m1_windows.pkl")
names = list(np.load("E:/tfb_nn/nn_data_afdem/names.npy", allow_pickle=True)); ci = names.index("last_queue")
lqz = np.load("E:/tfb_nn/nn_data_afdem_lastq.npz"); LQ, OFF = lqz["lq"], lqz["off"]
dt = "E:/tfb_nn/nn_test_afdem/"; Xt = np.load(dt + "X.npy", mmap_mode="r"); offt = np.load(dt + "off.npy")
wid = {w: i for i, w in enumerate(np.load(dt + "window_id.npy", allow_pickle=True).astype(str))}


def conf(Qm, obs, t, U, lq):
    d0 = t - t % 288
    if t - d0 < 6:
        return np.nan, np.nan, np.nan
    rows = np.arange(d0, t); b = (rows - d0) // 6
    O = Qm[rows]; ob = obs[rows]; Ub = U[b]
    far = np.ones(Qm.shape[1], bool)
    for j in np.flatnonzero(lq):
        far[max(0, j - 3):j + 4] = False
    num = (O * Ub).sum(); den = (Ub * ob).sum(); nO = O.sum()
    numf = (O * Ub)[:, far].sum(); denf = (Ub * ob)[:, far].sum()
    return num / den if den > 0 else np.nan, num / nO if nO > 0 else np.nan, numf / denf if denf > 0 else np.nan


out = []
for p in T2_PANELS:
    P = load(p, "train"); D = P.T // 288; n = P.n
    days = [str(x) for x in P.day]; mon = np.array([d[:7] for d in days]); wk = np.array([pd.Timestamp(d).dayofweek >= 5 for d in days])
    Qm = queue_matrix(P); obs = np.isfinite(P.V) & P.E
    qb = Qm.reshape(D, 48, 6, n).any(axis=2)
    U = {}
    for m in sorted(set(mon)) + ["none"]:
        for we in (False, True):
            U[(m, we)] = qb[(mon != m) & (wk == we)].mean(axis=0)
    for r in W[W.panel == p].itertuples():
        R, Pp, Rf = conf(Qm, obs, r.t, U[(r.month, r.weekend)], LQ[OFF[r.k]:OFF[r.k + 1]])
        out.append(dict(k=r.k, split="train", R_day=R, P_day=Pp, R_far=Rf))
    for split in ("validation", "private"):
        Pt = load(p, split); Qt = queue_matrix(Pt); ot = np.isfinite(Pt.V) & Pt.E; ti = {s: i for i, s in enumerate(Pt.times)}
        wi = pd.read_csv(REL / "task2" / p / split / "window_index.csv")
        for r in wi[wi.condition == "queue_ongoing"].itertuples():
            t = ti[r.forecast_origin]; i = wid[r.window_id]; we = pd.Timestamp(r.date).dayofweek >= 5
            lq = np.asarray(Xt[offt[i]:offt[i + 1], ci]) == 1
            R, Pp, Rf = conf(Qt, ot, t, U[("none", we)], lq)
            out.append(dict(window_id=r.window_id, split=split, R_day=R, P_day=Pp, R_far=Rf))
    load.cache_clear(); print(p, flush=True)
pd.DataFrame(out).to_pickle(r"C:/Users/user/tfb/review/m1_dayconf.pkl")
