"""March instrument M3, minimal version (PLAN_20261022 §2, INSTRUCTIONS_WORKER (ii), pre-registered in STATE.md 10/22 10:20).
Pseudo windows on the validation month's released (masked) data with the selector's conditions:
  onset   : no observed queued cell in T-12..T+5, one at T+6 (same day)
  ongoing : the history T-12..T-1 holds >= 2 queued obs with >= 2 on one link, a queue in T+1..T+6, persistence IoU
            <= 0.9 on observed cells; 6 h spacing (random order, seed 0) like t2_windows.candidates
  both    : history coverage (observed eligible cells / (12 n)) >= 0.5 (the month's normal coverage is ~0.57; official
            windows' histories are exempt from the Task 1 mask, trap 1 -> report by coverage band), and no row of
            T-12..T+6 inside an official window's blank (its T+1..T+18 rows)
Features exactly as t2_nn_af.py builds nn_test_afdem (TFB_T2_NOT=1, TFB_T2_ALLFIN=1, priors zeroed). Labels = the queue
matrix on observed eligible cells; the scoring mask = observed eligible cells only (trap 2). Each window gets k = the
number of earlier same-month days its evidence may use, drawn from the official distribution (0: 19%, 1: 20%,
2-3: 40%, 4-6: 21%; trap 3). Per panel, all onset-event origins (no coverage threshold) are stored for trap 4.
Outputs: NN/nn_test_m3/{X,off,window_id,cond}.npy (t2_nn_cpu_infer.py input) and work/t2/m3_meta.pkl.
usage: python t2_m3_build.py"""
import os, time, pickle, warnings, numpy as np, pandas as pd
os.environ["TFB_T2_NOT"] = "1"; os.environ["TFB_T2_ALLFIN"] = "1"; warnings.filterwarnings("ignore")
from cachelib import load, T2_PANELS
from common import REL, NN
from t2_model import PanelInputs
from t2_cache import window_X
from t2_arrival import arrival_features, _dmin
from t2_demand import demand_features, profile
from t2_windows import queue_matrix

H, F, COV = 12, 6, 0.5
VER = os.environ.get("TFB_M3VER", "")                   # 10/23 instrument v3: "v3" -> ongoing every 2 h (INSTRUMENT_V3_SPEC §2.5)
OGSP = 24 if VER == "v3" else 72                          # ongoing spacing in rows (v2: 6 h, as the selector)
KV, KP = np.arange(7), np.array([0.19, 0.20, 0.20, 0.20, 0.07, 0.07, 0.07])
names = [str(x) for x in np.load(f"{NN}/nn_data_afdem/names.npy", allow_pickle=True)]
I_PON, I_POG = names.index("prior_on"), names.index("prior_og")
rng_k = np.random.default_rng(20261022)
t0 = time.time(); rows, wid, cond, meta, events = [], [], [], [], {}

for p in T2_PANELS:
    P = load(p, "validation"); PI = PanelInputs(P); prof = profile(load(p, "train"))
    ti = {s: i for i, s in enumerate(P.times)}; T = P.T; n = P.n
    Qm = queue_matrix(P); obs = np.isfinite(P.V) & P.E
    blank = np.zeros(T, bool)
    for r in pd.read_csv(REL / "task2" / p / "validation" / "window_index.csv").itertuples():
        t0w = ti[r.forecast_origin]; blank[t0w + 1:t0w + 19] = True
    any_q = Qm.any(axis=1); qrow = np.concatenate([[0], np.cumsum(any_q)])
    cnt = np.concatenate([np.zeros((1, n), np.int32), np.cumsum(Qm, axis=0, dtype=np.int32)])
    ocnt = np.concatenate([[0], np.cumsum(obs.sum(axis=1))])
    ons, ogs, ev = [], [], []
    for t in range(H, T - F):
        if t // 288 != (t + F) // 288 or (t - H) // 288 != t // 288:
            continue
        is_on = qrow[t + F] - qrow[t - H] == 0 and any_q[t + F]
        if is_on:
            ev.append(t)                                         # trap 4: every onset event, any coverage
        cov = (ocnt[t] - ocnt[t - H]) / (H * n)
        if cov < COV or blank[t - H:t + F + 1].any():
            continue
        if is_on:
            ons.append((t, cov))
            continue
        hq = cnt[t] - cnt[t - H]
        if hq.sum() >= 2 and hq.max() >= 2 and qrow[t + F + 1] - qrow[t + 1] > 0:
            fut = Qm[t + 1:t + F + 1]; el = obs[t + 1:t + F + 1]; last = np.broadcast_to(Qm[t - 1], fut.shape)
            u = ((last | fut) & el).sum()
            if (1.0 if u == 0 else (last & fut & el).sum() / u) <= 0.9:
                ogs.append((t, cov))
    chosen = []
    for i in np.random.default_rng(0).permutation(len(ogs)):    # 6 h spacing for ongoing, as the selector
        if all(abs(ogs[i][0] - o[0]) >= OGSP for o in chosen):
            chosen.append(ogs[i])
    events[p] = np.array(ev, int)
    for c, lst in (("queue_onset", ons), ("queue_ongoing", sorted(chosen))):
        for t, cov in lst:
            Xw, _ = window_X(PI, t, c); Xw[:, I_PON] = 0.0; Xw[:, I_POG] = 0.0
            V, Q, tl = PI.window(t); Xa, _ = arrival_features(P, V, Q, tl, PI.x)
            Xd = demand_features(P, Q, tl, t, P.times[t][:10], prof, _dmin(P, PI.x))
            rows.append(np.column_stack([Xw, Xa, Xd]).astype(np.float32)); w = f"M3_{p}_{t}"; wid.append(w); cond.append(c)
            meta.append(dict(window_id=w, panel=p, t=t, cond=c, cov=float(cov), day=t // 288, slot=t % 288,
                             morning=bool((t + F) % 288 < 144), k=int(rng_k.choice(KV, p=KP)),
                             y=Qm[t + 1:t + F + 1].T.copy(), m=obs[t + 1:t + F + 1].T.copy()))
    print(p, "onset", len(ons), "ongoing", len(chosen), "(candidates", len(ogs), ") onset events", len(ev), f"{time.time() - t0:.0f}s", flush=True)
    load.cache_clear()

d = f"{NN}/nn_test_m3{VER}"; os.makedirs(d, exist_ok=True)
off = np.concatenate([[0], np.cumsum([len(r) for r in rows])])
np.save(f"{d}/X.npy", np.vstack(rows)); np.save(f"{d}/off.npy", off)
np.save(f"{d}/window_id.npy", np.array(wid)); np.save(f"{d}/cond.npy", np.array(cond))
pickle.dump(dict(meta=meta, events=events, cov_threshold=COV, k_dist=dict(zip(KV.tolist(), KP.tolist()))),
            open(f"C:/Users/user/tfb/work/t2/m3{VER}_meta.pkl", "wb"))
M = pd.DataFrame([{k: v for k, v in m.items() if k not in ("y", "m")} for m in meta])
M["band"] = pd.cut(M["cov"], [0.5, 0.6, 0.7, 1.01], right=False, labels=["0.5-0.6", "0.6-0.7", ">=0.7"])
print(M.groupby(["cond", "band"]).size().unstack(fill_value=0).to_string())
print("morning onset", int((M.morning & (M.cond == "queue_onset")).sum()), "| windows", len(M), "| rows", off[-1], f"| {time.time() - t0:.0f}s")
