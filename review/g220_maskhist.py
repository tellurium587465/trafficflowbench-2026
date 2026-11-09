"""G220 (11/02, ledger #188): is the row-T gain a stand-in for a masked history? January and February selector windows (the OOF of the fold-2 networks): inputs rebuilt as the official windows have them (history unmasked) and as M3 has them (the day's Task 1 mask also on T-12 .. T-1, TFB_T2_MASKHIST=1), each with and without the row at T; capB0 and capBT seeds 0-3 on the CPU; raw best_set IoU against the labels. Identity first: the unmasked rebuild against the stored nn_data_afdem rows.
usage: py -B -X utf8 g220_maskhist.py build | read"""
import os, sys, pickle, shutil, subprocess, numpy as np, pandas as pd, warnings
MODE = sys.argv[1]; warnings.filterwarnings("ignore"); sys.dont_write_bytecode = True; sys.path.insert(0, r"C:/Users/user/tfb/src")
NN = "C:/tfb_data/tfb_nn/"; d = NN + "nn_data_afdem/"; date = np.load(d + "date.npy", allow_pickle=True); cond = np.load(d + "cond.npy", allow_pickle=True); isel = np.load(d + "is_sel.npy"); pan = np.array([str(x) for x in np.load(d + "panel.npy", allow_pickle=True)]); t_ = np.load(d + "t.npy"); off = np.load(d + "off.npy")
mon = np.array([str(x)[:7] for x in date]); KS = np.flatnonzero(isel & np.isin(mon, ("2031-01", "2031-02")) & (cond == "queue_ongoing"))
if MODE == "build":
    for mh in ("0", "1"):
        os.environ["TFB_T2_NOT"] = "1"; os.environ["TFB_T2_ALLFIN"] = "1"; os.environ["TFB_T2_MASKHIST"] = mh
        code = f'''
import sys, numpy as np, warnings; warnings.filterwarnings("ignore"); sys.path.insert(0, r"C:/Users/user/tfb/src")
from cachelib import load
from t2_model import PanelInputs
from t2_cache import window_X
from t2_arrival import arrival_features, _dmin
from t2_demand import demand_features, profile
KS = np.array({KS.tolist()}); pan = {pan[KS].tolist()}; t_ = {t_[KS].tolist()}; names = [str(x) for x in np.load("{d}names.npy", allow_pickle=True)]; iv, iq = names.index("v_T"), names.index("q_T")
XS, XT, OF = [], [], [0]; PIc = {{}}
for k, p, t in zip(KS, pan, t_):
    if p not in PIc:
        P = load(p, "train"); PIc[p] = (P, PanelInputs(P), profile(P))
    P, PI, prof = PIc[p]; Xw, _ = window_X(PI, t, "queue_ongoing"); Xw[:, names.index("prior_on")] = 0.0; Xw[:, names.index("prior_og")] = 0.0; V, Q, tl = PI.window(t); Xa, _ = arrival_features(P, V, Q, tl, PI.x); Xd = demand_features(P, Q, tl, t, P.times[t][:10], prof, _dmin(P, PI.x)); X = np.column_stack([Xw, Xa, Xd]).astype(np.float32); XS.append(X); X2 = X.copy(); X2[:, iv] = PI.mask_v[t] / P.vf; X2[:, iq] = PI.mask_q[t] / P.cap; XT.append(X2); OF.append(OF[-1] + len(X))
import os
for nm, XX in (("mh{mh}", XS), ("mh{mh}t", XT)):
    o = "{NN}nn_test_jf" + nm + "/"; os.makedirs(o, exist_ok=True); np.save(o + "X.npy", np.vstack(XX)); np.save(o + "off.npy", np.array(OF)); np.save(o + "window_id.npy", np.array([str(k) for k in KS])); np.save(o + "cond.npy", np.array(["queue_ongoing"] * len(KS)))
print("built", "{mh}", len(KS), flush=True)
'''
        subprocess.run([sys.executable, "-B", "-c", code], check=True, env=dict(os.environ))
    X0 = np.load(NN + "nn_test_jfmh0/X.npy"); o0 = np.load(NN + "nn_test_jfmh0/off.npy"); Xs = np.load(d + "X.npy", mmap_mode="r"); dm = []
    for i, k in enumerate(KS[:40]):
        a = np.asarray(Xs[off[k]:off[k + 1]]); b = X0[o0[i]:o0[i + 1]]; dm.append(float(np.nanmax(np.abs(np.where(np.isnan(a), 0, a) - np.where(np.isnan(b), 0, b)))) if a.shape == b.shape else 1e9)
    print("identity, unmasked rebuild against nn_data_afdem on 40 windows: max |d|", max(dm), flush=True); sys.exit()
os.environ["TFB_M3VER"] = "v3"
src = open(r"C:/Users/user/tfb/review/g119_month_postproc.py", encoding="utf-8").read().split('if __name__ == "__main__":')[0]
sys.argv = [sys.argv[0], "run"]; exec(compile(src, "g119", "exec"), globals())
R_ = {}
for dn in ("mh0", "mh0t", "mh1", "mh1t"):
    for tag in ("capB0", "capBT"):
        pr = [pickle.load(open(W + f"probs_jf{dn}__{tag}_s{s}_fold2030-12.pkl", "rb")) for s in range(4)]; R_[(dn, tag)] = {k: sum(x[k] for x in pr) / 4 for k in pr[0]}
rows = []
for k in KS:
    k = int(k); lab, el = T.LAB[k]; y, e = lab.astype(bool), el.astype(bool); r = dict(w=k, month=MON[k], panel=str(T.META["panel"][k]), day=int(str(T.META["date"][k])[8:10]))
    for (dn, tag), P_ in R_.items():
        r[f"{dn}_{tag}"] = iou(decode(P_[str(k)]), y, e, "queue_ongoing")
    rows.append(r)
D = pd.DataFrame(rows)


def clm(v, n_boot=2000):
    ids = pd.factorize(D.panel + "_" + D.month + "_" + D.day.astype(str))[0]; a = np.asarray(v, float); M = np.bincount(ids, a); N = np.bincount(ids); Rm = np.random.default_rng(0).multinomial(len(M), np.full(len(M), 1 / len(M)), size=n_boot); return a.mean(), float(((Rm @ M) / (Rm @ N)).std())


print("windows", len(D))
for nm, a, b in (("(b) history unmasked, row T: capBT(T) - capB0", "mh0_capB0", "mh0t_capBT"), ("(a) history masked as M3, row T: capBT(T) - capB0", "mh1_capB0", "mh1t_capBT"), ("unmasked, capBT without the row T - capB0", "mh0_capB0", "mh0_capBT"), ("masked, history mask cost on capB0", "mh0_capB0", "mh1_capB0")):
    m, s = clm(D[b] - D[a]); print(f"{nm}: {m:+.4f} (SE {s:.4f}, z {m / s:+.2f})", flush=True)
