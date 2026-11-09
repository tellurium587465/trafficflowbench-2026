"""G203 (11/02, ledger #167): same-month evidence as network inputs (q27 / q28 again, on the right instrument). Three columns per link: ev_on = a queue seen within +-2 links in rows -30 .. +60 min of the same clock on the earlier days, ev_og = the same within +-1 link in rows -90 .. +60 min, ev_nd = earlier days used / 3. Train: the Task 1 masked layer, earlier days of the same month, k drawn per window from m3v2.kdist(day of month, cap 3) (seed 167), no simulated blanks. March: m3v2.seen at the most likely k of the window (with its simulated blanks).
Writes C:/tfb_data/tfb_nn/trow/trow_ev.npy = (rows of X, 5): v_T, q_T, ev_on, ev_og, ev_nd, and C:/tfb_data/tfb_nn/nn_test_m3v3e/ (X with row T filled and the 3 columns appended).
usage: TFB_M3VER=v3 py -B -X utf8 g203_ev_build.py"""
import os, sys, pickle, shutil, numpy as np, warnings
os.environ["TFB_T2_NOT"] = "1"; os.environ["TFB_T2_ALLFIN"] = "1"; os.environ.setdefault("TFB_M3VER", "v3"); warnings.filterwarnings("ignore"); sys.dont_write_bytecode = True; sys.path.insert(0, r"C:/Users/user/tfb/src")
from cachelib import load
from t2_model import PanelInputs
import m3v2
NN = "C:/tfb_data/tfb_nn/"; O = NN + "trow/"; d = NN + "nn_data_afdem/"; off = np.load(d + "off.npy"); pan = np.array([str(x) for x in np.load(d + "panel.npy", allow_pickle=True)]); t_ = np.load(d + "t.npy"); TR = np.load(O + "trow_train.npy"); E = np.zeros((TR.shape[0], 3), np.float32); rng = np.random.default_rng(167)


def dil(s, r):
    x = s.copy()
    for o in range(1, r + 1):
        x[o:] |= s[:-o]; x[:-o] |= s[o:]
    return x


for p in sorted(set(pan)):
    P = load(p, "train"); PI = PanelInputs(P); q = (PI.mask_v <= P.vcut[None, :]) & np.isfinite(PI.mask_v); mon = np.array([str(x)[:7] for x in P.times]); r0 = {m: int(np.flatnonzero(mon == m)[0]) for m in sorted(set(mon))}; ks = np.flatnonzero(pan == p)
    for k in ks:
        t = int(t_[k]); dom = (t - r0[mon[t]]) // 288; kd = m3v2.kdist(dom, cap=3); kk = int(rng.choice(list(kd), p=np.array(list(kd.values())) / sum(kd.values()))); s1 = np.zeros(P.n, bool); s2 = s1.copy(); nd = 0
        for back in range(1, kk + 1):
            dd = t - back * 288
            if dd - 18 >= 0:
                s1 |= q[dd - 6:dd + 12].any(axis=0); s2 |= q[dd - 18:dd + 12].any(axis=0); nd += 1
        a, b = int(off[k]), int(off[k + 1]); E[a:b, 0] = dil(s1, 2); E[a:b, 1] = dil(s2, 1); E[a:b, 2] = nd / 3
    print(p, len(ks), flush=True); load.cache_clear()
np.save(O + "trow_ev.npy", np.hstack([TR, E])); print("train: ev_on share", round(float(E[:, 0].mean()), 4), "ev_og", round(float(E[:, 1].mean()), 4), "nd", round(float(E[:, 2].mean()) * 3, 3), flush=True)
m = pickle.load(open("C:/Users/user/tfb/work/t2/m3v3_meta.pkl", "rb"))["meta"]; M = {x["window_id"]: x for x in m}; d2 = NN + "nn_test_m3v3t/"; D2 = NN + "nn_test_m3v3e/"; os.makedirs(D2, exist_ok=True)
for f in ("cond", "off", "window_id", "R"):
    if not os.path.exists(D2 + f + ".npy"):
        shutil.copy(d2 + f + ".npy", D2 + f + ".npy")
X2 = np.load(d2 + "X.npy"); off2 = np.load(d2 + "off.npy"); wid = [str(w) for w in np.load(d2 + "window_id.npy", allow_pickle=True)]; E2 = np.zeros((X2.shape[0], 3), np.float32)
for i, w in enumerate(wid):
    x = M[w]; kd = m3v2.kdist(int(x["day"]), cap=3); kk = max(kd, key=kd.get); a, b = int(off2[i]), int(off2[i + 1]); s1, nd = m3v2.seen(x["panel"], int(x["t"]), kk, -6, 12); s2, _ = m3v2.seen(x["panel"], int(x["t"]), kk, -18, 12); E2[a:b, 0] = dil(s1, 2); E2[a:b, 1] = dil(s2, 1); E2[a:b, 2] = nd / 3
np.save(D2 + "X.npy", np.hstack([X2, E2])); print("M3: ev_on share", round(float(E2[:, 0].mean()), 4), "ev_og", round(float(E2[:, 1].mean()), 4), "nd", round(float(E2[:, 2].mean()) * 3, 3), flush=True)
