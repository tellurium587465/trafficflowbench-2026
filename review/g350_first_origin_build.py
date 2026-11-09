"""G350 (11/05, ledger #279 stage 1 data): training windows at the FIRST origin of each run of qualifying origins (t qualifies for the
condition and t-1 does not), the official selector's choice (no spacing limit), both conditions, training months 2030-09 .. 2031-02 (cvf
fold 2030-06 left out 06 .. 08). The condition per origin is t2_windows.candidates' (same calendar day T-12 .. T+6, observed coverage of
T-12 .. T-1 >= 0.7, a queue in T+1 .. T+6; onset = no queue in T-12 .. T+5 and a queue at T+6; ongoing = >= 2 queued history cells with a
link queued >= 2 rows and IoU(persistence of T-1, T+1 .. T+6 on P.E) <= 0.9). Inputs as nn_data_afdem (window_X + arrival + demand,
TFB_T2_NOT=1, TFB_T2_ALLFIN=1, priors 0), Y = queue_matrix rows t+1 .. t+6, EL = P.E rows t+1 .. t+6 -> NN/nn_data_first (train layout:
X, Y, EL, off, date, cond, panel, t). usage: py -B -X utf8 g350_first_origin_build.py [count]"""
import os, sys, time
os.environ["TFB_T2_NOT"] = "1"; os.environ["TFB_T2_ALLFIN"] = "1"
import numpy as np
sys.path.insert(0, r"C:/Users/user/tfb/src")
from cachelib import load, T2_PANELS
from t2_windows import queue_matrix, iou, H, F
T0 = time.time(); ONLY_COUNT = (sys.argv[1:] or [""])[0] == "count"; NN = "C:/tfb_data/tfb_nn/"; MONTHS = ("2030-09", "2030-10", "2030-11", "2030-12", "2031-01", "2031-02")


def qualify(P, Qm):
    obs = np.isfinite(P.V) & P.E; any_q = Qm.any(axis=1); cnt = np.concatenate([np.zeros((1, P.n), np.int32), np.cumsum(Qm, axis=0, dtype=np.int32)])
    ocnt = np.concatenate([[0], np.cumsum(obs.sum(axis=1))]); qrow = np.concatenate([[0], np.cumsum(any_q)]); on = np.zeros(P.T, bool); og = np.zeros(P.T, bool)
    for t in range(H, P.T - F):
        if P.times[t - H][:10] != P.times[t + F][:10] or (ocnt[t] - ocnt[t - H]) / (H * P.n) < 0.7 or not qrow[t + F + 1] - qrow[t + 1] > 0:
            continue
        if qrow[t + F] - qrow[t - H] == 0 and any_q[t + F]:
            on[t] = True
        else:
            hq = cnt[t] - cnt[t - H]
            if hq.sum() >= 2 and hq.max() >= 2:
                fut = Qm[t + 1:t + F + 1]; el = P.E[t + 1:t + F + 1]
                og[t] = iou(np.broadcast_to(Qm[t - 1], fut.shape), fut, el) <= 0.9
    return on, og


items = []
for p in T2_PANELS:
    P = load(p, "train"); Qm = queue_matrix(P); on, og = qualify(P, Qm); mon = np.array([s[:7] for s in P.times]); inm = np.isin(mon, MONTHS)
    for cond, q in (("queue_onset", on), ("queue_ongoing", og)):
        first = q & ~np.concatenate([[False], q[:-1]]) & inm
        items += [(p, int(t), cond) for t in np.flatnonzero(first)]
        print(f"{p} {cond}: qualifying origins {int((q & inm).sum())}, runs (first origins) {int(first.sum())} ({time.time() - T0:.0f}s)", flush=True)
    load.cache_clear()
print(f"first-origin windows: {len(items)} (onset {sum(c == 'queue_onset' for _, _, c in items)}, ongoing {sum(c == 'queue_ongoing' for _, _, c in items)}), {time.time() - T0:.0f}s", flush=True)
if ONLY_COUNT:
    sys.exit()
from t2_model import PanelInputs
from t2_cache import window_X
from t2_arrival import arrival_features, _dmin
from t2_demand import demand_features, profile
names = [str(x) for x in np.load(NN + "nn_data_afdem/names.npy", allow_pickle=True)]; XS, YS, ES, OF, DT, CD, PN, TT = [], [], [], [0], [], [], [], []
for p in T2_PANELS:
    its = [x for x in items if x[0] == p]
    if not its:
        continue
    P = load(p, "train"); PI = PanelInputs(P); prof = profile(P); Qm = queue_matrix(P)
    for _, t, cd in its:
        Xw, _ = window_X(PI, t, cd); Xw[:, names.index("prior_on")] = 0.0; Xw[:, names.index("prior_og")] = 0.0; V, Q, tl = PI.window(t); Xa, _ = arrival_features(P, V, Q, tl, PI.x); Xd = demand_features(P, Q, tl, t, P.times[t][:10], prof, _dmin(P, PI.x))
        X = np.column_stack([Xw, Xa, Xd]).astype(np.float32); XS.append(X); YS.append(Qm[t + 1:t + 7].T.astype(np.uint8)); ES.append(P.E[t + 1:t + 7].T.astype(np.uint8)); OF.append(OF[-1] + len(X))
        DT.append(P.times[t][:10]); CD.append(cd); PN.append(p); TT.append(t)
    load.cache_clear(); print(f"{p} built ({time.time() - T0:.0f}s)", flush=True)
o = NN + "nn_data_first/"; os.makedirs(o, exist_ok=True)
for f, v in (("X", np.vstack(XS)), ("Y", np.vstack(YS)), ("EL", np.vstack(ES)), ("off", np.array(OF)), ("date", np.array(DT)), ("cond", np.array(CD)), ("panel", np.array(PN)), ("t", np.array(TT))):
    np.save(o + f + ".npy", v)
print(f"nn_data_first: {len(TT)} windows, {OF[-1]} rows, {time.time() - T0:.0f}s", flush=True)
