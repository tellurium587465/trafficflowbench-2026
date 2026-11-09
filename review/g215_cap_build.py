"""G215 (11/02, ledger #180b): the month's breakdown flow as network inputs, on top of the row at T. Per link q_bd = the median over the births on the earlier days (k days, +-1 link) of the flow in the 15 minutes before the birth (a link's first queued row after three clear rows, Task 1 masked layer on train, the release on March). Four columns: q(T - 5) / q_bd, q(T - 15) / q_bd, (q(T - 5) - q(T - 15)) / q_bd, (q_bd - q(T - 5)) / cap; NaN where there is no birth (the network's missing mask). Train: k drawn per window from m3v2.kdist(day, cap 3), seed 180. March: the most likely k.
Writes C:/tfb_data/tfb_nn/trow/trow_cap.npy = (rows of X, 6): v_T, q_T, four columns, and C:/tfb_data/tfb_nn/nn_test_m3v3c/.
usage: TFB_M3VER=v3 py -B -X utf8 g215_cap_build.py"""
import os, sys, pickle, shutil, numpy as np, warnings
os.environ["TFB_T2_NOT"] = "1"; os.environ["TFB_T2_ALLFIN"] = "1"; os.environ.setdefault("TFB_M3VER", "v3"); warnings.filterwarnings("ignore"); sys.dont_write_bytecode = True; sys.path.insert(0, r"C:/Users/user/tfb/src")
from cachelib import load
from t2_model import PanelInputs
import m3v2
NN = "C:/tfb_data/tfb_nn/"; O = NN + "trow/"; d = NN + "nn_data_afdem/"; off = np.load(d + "off.npy"); pan = np.array([str(x) for x in np.load(d + "panel.npy", allow_pickle=True)]); t_ = np.load(d + "t.npy"); TR = np.load(O + "trow_train.npy"); E = np.full((TR.shape[0], 4), np.nan, np.float32); rng = np.random.default_rng(180)


def births(P):
    PI = PanelInputs(P); V, Q = PI.mask_v, PI.mask_q; q = (V <= P.vcut[None, :]) & np.isfinite(V); b = q[3:] & ~q[2:-1] & ~q[1:-2] & ~q[:-3]; r, j = np.nonzero(b); r = r + 3
    with np.errstate(all="ignore"):
        Qc = np.nan_to_num(Q, nan=0.0); Fc = np.isfinite(Q).astype(float); qb = (Qc[r - 3, j] + Qc[r - 2, j] + Qc[r - 1, j]) / np.maximum(Fc[r - 3, j] + Fc[r - 2, j] + Fc[r - 1, j], 1)
    ok = (Fc[r - 3, j] + Fc[r - 2, j] + Fc[r - 1, j]) > 0; return r[ok], j[ok], qb[ok], Q


def cols(B, P, t, d0, d1):
    r, j, qb, Q = B; a = np.searchsorted(r, d0); b = np.searchsorted(r, d1); bd = np.full(P.n, np.nan)
    if b > a:
        jj, vv = j[a:b], qb[a:b]
        for c_ in {int(x) + o for x in np.unique(jj) for o in (-1, 0, 1)}:
            if 0 <= c_ < P.n:
                bd[c_] = np.median(vv[np.abs(jj - c_) <= 1])
    q1 = Q[t - 1]; q4 = Q[t - 4]
    with np.errstate(all="ignore"):
        return np.column_stack([q1 / bd, q4 / bd, (q1 - q4) / bd, (bd - q1) / P.cap])


for p in sorted(set(pan)):
    P = load(p, "train"); B = births(P); mon = np.array([str(x)[:7] for x in P.times]); r0 = {m: int(np.flatnonzero(mon == m)[0]) for m in sorted(set(mon))}; ks = np.flatnonzero(pan == p)
    for k in ks:
        t = int(t_[k]); m0 = r0[mon[t]]; dom = (t - m0) // 288; kd = m3v2.kdist(dom, cap=3); kk = int(rng.choice(list(kd), p=np.array(list(kd.values())) / sum(kd.values()))); dn = t // 288 * 288; a, b = int(off[k]), int(off[k + 1]); E[a:b] = cols(B, P, t, max(m0, dn - kk * 288), dn)
    print(p, len(ks), flush=True); load.cache_clear()
np.save(O + "trow_cap.npy", np.hstack([TR, E])); print("train: share with q_bd", round(float(np.isfinite(E[:, 0]).mean()), 4), flush=True)
m = pickle.load(open("C:/Users/user/tfb/work/t2/m3v3_meta.pkl", "rb"))["meta"]; M = {x["window_id"]: x for x in m}; d2 = NN + "nn_test_m3v3t/"; D2 = NN + "nn_test_m3v3c/"; os.makedirs(D2, exist_ok=True)
for f in ("cond", "off", "window_id", "R"):
    if not os.path.exists(D2 + f + ".npy"):
        shutil.copy(d2 + f + ".npy", D2 + f + ".npy")
X2 = np.load(d2 + "X.npy"); off2 = np.load(d2 + "off.npy"); wid = [str(w) for w in np.load(d2 + "window_id.npy", allow_pickle=True)]; E2 = np.full((X2.shape[0], 4), np.nan, np.float32); BB = {}
for i, w in enumerate(wid):
    x = M[w]; p = x["panel"]
    if p not in BB:
        BB[p] = (load(p, "validation"),); BB[p] = (BB[p][0], births(BB[p][0]))
    P, B = BB[p]; t = int(x["t"]); kd = m3v2.kdist(int(x["day"]), cap=3); kk = max(kd, key=kd.get); dn = t // 288 * 288; a, b = int(off2[i]), int(off2[i + 1]); E2[a:b] = cols(B, P, t, max(0, dn - kk * 288), dn)
np.save(D2 + "X.npy", np.hstack([X2, E2])); print("M3: share with q_bd", round(float(np.isfinite(E2[:, 0]).mean()), 4), "windows", len(wid), flush=True)
