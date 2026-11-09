"""E4 (PLAN_20261022 §3.1, INSTRUCTIONS_WORKER (v)): March (the validation month) as a 10th training month, for the
April-window models only. Training windows are cut from the released masked March data with the training-pool rules of
build_nn2 (candidate origins every 30 min, both conditions, same day) with history coverage >= 0.5 (March's normal
coverage is ~0.57; the official windows' histories are exempt from the Task 1 mask), excluding any origin whose
T-12..T+6 rows touch an official window's blank (its T+1..T+18 rows). Features exactly as the production test features
(window_X + arrival + demand, TFB_T2_NOT=1, TFB_T2_ALLFIN=1, demand profiles from the train months). Labels = observed
queued cells; EL = observed eligible future cells only, so unobserved future cells carry no loss.
Writes NN/nn_data_march (train layout, dates 2031-03-xx, fold_month 2031-03).
10/22 (X1): the same cut for the private month (April) with split=private -> NN/nn_data_april; used ONLY to fine-tune on
the completed earlier days of the month before each window's day (t2_nn_finetune.py), never as a training month.
usage: python build_march.py [spacing=6] [split=validation]"""
import os, sys, time, numpy as np, pandas as pd
os.environ["TFB_T2_NOT"] = "1"; os.environ["TFB_T2_ALLFIN"] = "1"
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from cachelib import load, T2_PANELS
from common import REL, NN
from t2_model import PanelInputs
from t2_cache import window_X
from t2_arrival import arrival_features, _dmin
from t2_demand import demand_features, profile
from t2_windows import queue_matrix

H, F, COV = 12, 6, 0.5
SP = int(sys.argv[1]) if len(sys.argv) > 1 else 6
SPLIT = sys.argv[2] if len(sys.argv) > 2 else "validation"; OUTN = os.environ.get("TFB_MONTH_OUT") or {"validation": "nn_data_march", "private": "nn_data_april"}[SPLIT]   # 10/23: TFB_MONTH_OUT for t2_window_entry.py
names = [str(x) for x in np.load(f"{NN}/nn_data_afdem/names.npy", allow_pickle=True)]
t0 = time.time(); rows, meta, Y, EL = [], [], [], []
for p in T2_PANELS:
    P = load(p, SPLIT); PI = PanelInputs(P); prof = profile(load(p, "train")); ti = {s: i for i, s in enumerate(P.times)}
    Qm = queue_matrix(P); obs = np.isfinite(P.V) & P.E; n = P.n
    blank = np.zeros(P.T, bool)
    for r in pd.read_csv(REL / "task2" / p / SPLIT / "window_index.csv").itertuples():
        b0 = ti[r.forecast_origin]; blank[b0 + 1:b0 + 19] = True
    any_q = Qm.any(axis=1); qrow = np.concatenate([[0], np.cumsum(any_q)])
    cnt = np.concatenate([np.zeros((1, n), np.int32), np.cumsum(Qm, axis=0, dtype=np.int32)])
    ocnt = np.concatenate([[0], np.cumsum(obs.sum(axis=1))])
    last = {"queue_onset": -10 ** 9, "queue_ongoing": -10 ** 9}; k = 0
    for t in range(H, P.T - F):
        if (t - H) // 288 != (t + F) // 288 or blank[t - H:t + F + 1].any():
            continue
        if (ocnt[t] - ocnt[t - H]) / (H * n) < COV or not (qrow[t + F + 1] - qrow[t + 1] > 0):
            continue
        if qrow[t + F] - qrow[t - H] == 0 and any_q[t + F]:
            c = "queue_onset"
        else:
            hq = cnt[t] - cnt[t - H]
            if not (hq.sum() >= 2 and hq.max() >= 2):
                continue
            fut = Qm[t + 1:t + F + 1]; el = obs[t + 1:t + F + 1]; lastq = np.broadcast_to(Qm[t - 1], fut.shape)
            u = ((lastq | fut) & el).sum()
            if (1.0 if u == 0 else (lastq & fut & el).sum() / u) > 0.9:
                continue
            c = "queue_ongoing"
        if t - last[c] < SP:
            continue
        last[c] = t
        Xw, _ = window_X(PI, t, c)
        V, Q, tl = PI.window(t); Xa, _ = arrival_features(P, V, Q, tl, PI.x)
        Xd = demand_features(P, Q, tl, t, P.times[t][:10], prof, _dmin(P, PI.x))
        rows.append(np.column_stack([Xw, Xa, Xd]).astype(np.float32))
        Y.append((Qm[t + 1:t + F + 1] & obs[t + 1:t + F + 1]).T.astype(np.uint8)); EL.append(obs[t + 1:t + F + 1].T.astype(np.uint8))
        meta.append(dict(panel=p, t=t, cond=c, date=str(P.times[t])[:10])); k += 1
    print(p, SPLIT, "training windows", k, f"{time.time() - t0:.0f}s", flush=True)
    load.cache_clear()
d = f"{NN}/{OUTN}"; os.makedirs(d, exist_ok=True)
off = np.concatenate([[0], np.cumsum([len(r) for r in rows])])
X = np.lib.format.open_memmap(f"{d}/X.npy", mode="w+", dtype=np.float32, shape=(int(off[-1]), len(names)))
for i, r in enumerate(rows):
    X[off[i]:off[i + 1]] = r
X.flush(); del X
np.save(f"{d}/off.npy", off); np.save(f"{d}/names.npy", np.array(names))
for f in ("panel", "t", "cond", "date"):
    np.save(f"{d}/{f}.npy", np.array([m[f] for m in meta]))
np.save(f"{d}/is_eval.npy", np.zeros(len(meta), bool)); np.save(f"{d}/is_sel.npy", np.zeros(len(meta), bool))
np.save(f"{d}/Y.npy", np.vstack(Y)); np.save(f"{d}/EL.npy", np.vstack(EL))
c = pd.Series([m["cond"] for m in meta]).value_counts().to_dict()
print(OUTN, len(meta), "windows", c, off[-1], "rows", f"{time.time() - t0:.0f}s")
