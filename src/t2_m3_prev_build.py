"""E3 support (pre-registered STATE.md 10/22 11:00): for every M3 window at origin T (day d, evidence limit k), the
pseudo-origins T' = T - b*288 for b = 1..min(k, d) at the same clock time on earlier days of the month. For each T':
production features (as t2_m3_build) for the model's own prediction there, and the observed outcome at T'+1..T'+6
(queued, observed) with trap 4 applied (onset events within +-60 min of T' on that day hide rows e+1..e+18 -> unobserved).
Outputs NN/nn_test_m3prev (t2_nn_cpu_infer.py input; ids <M3 window>_b<b>) and work/t2/m3prev_meta.pkl.
usage: python t2_m3_prev_build.py"""
import os, pickle, time, warnings, numpy as np
os.environ["TFB_T2_NOT"] = "1"; os.environ["TFB_T2_ALLFIN"] = "1"; warnings.filterwarnings("ignore")
from cachelib import load
from common import NN
from t2_model import PanelInputs
from t2_cache import window_X
from t2_arrival import arrival_features, _dmin
from t2_demand import demand_features, profile
from t2_windows import queue_matrix

H, F = 12, 6
names = [str(x) for x in np.load(f"{NN}/nn_data_afdem/names.npy", allow_pickle=True)]
I_PON, I_POG = names.index("prior_on"), names.index("prior_og")
M = pickle.load(open("C:/Users/user/tfb/work/t2/m3_meta.pkl", "rb")); EV = M["events"]
t0 = time.time(); rows, wid, cond, meta = [], [], [], []
by_panel = {}
for m in M["meta"]:
    by_panel.setdefault(m["panel"], []).append(m)
for p, ms in by_panel.items():
    P = load(p, "validation"); PI = PanelInputs(P); prof = profile(load(p, "train"))
    Qm = queue_matrix(P); obs = np.isfinite(P.V) & P.E
    for m in ms:
        t, d, k = m["t"], m["day"], m["k"]
        for b in range(1, min(k, d) + 1):
            tp = t - b * 288
            if tp < H:
                break
            hid = np.zeros(F, bool)
            for e in EV[p][np.abs(EV[p] - tp) <= 12]:                   # trap 4 on that earlier day
                for h in range(F):
                    if e + 1 <= tp + 1 + h <= e + 18:
                        hid[h] = True
            Xw, _ = window_X(PI, tp, m["cond"]); Xw[:, I_PON] = 0.0; Xw[:, I_POG] = 0.0
            V, Q, tl = PI.window(tp); Xa, _ = arrival_features(P, V, Q, tl, PI.x)
            Xd = demand_features(P, Q, tl, tp, P.times[tp][:10], prof, _dmin(P, PI.x))
            rows.append(np.column_stack([Xw, Xa, Xd]).astype(np.float32)); w = f"{m['window_id']}_b{b}"; wid.append(w); cond.append(m["cond"])
            mo = obs[tp + 1:tp + F + 1].copy(); mo[hid] = False
            meta.append(dict(window_id=w, parent=m["window_id"], b=b, y=(Qm[tp + 1:tp + F + 1] & mo).T.copy(), m=mo.T.copy()))
    print(p, "prev origins", sum(1 for x in meta if x["parent"].startswith(f"M3_{p}_")), f"{time.time() - t0:.0f}s", flush=True)
    load.cache_clear()
d = f"{NN}/nn_test_m3prev"; os.makedirs(d, exist_ok=True)
off = np.concatenate([[0], np.cumsum([len(r) for r in rows])])
np.save(f"{d}/X.npy", np.vstack(rows)); np.save(f"{d}/off.npy", off)
np.save(f"{d}/window_id.npy", np.array(wid)); np.save(f"{d}/cond.npy", np.array(cond))
pickle.dump(meta, open("C:/Users/user/tfb/work/t2/m3prev_meta.pkl", "wb"))
print("prev origins", len(meta), "rows", off[-1], f"{time.time() - t0:.0f}s")
