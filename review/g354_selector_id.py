"""G354 (11/05, ledger #280, measurement only): identify the official selector on the training split's June 2030 (80 windows). Per panel,
per variant of the qualification (t2_windows.candidates' rules: same day T-12 .. T+6, coverage of T-12 .. T-1 >= 0.7, a queue in T+1 .. T+6;
onset = no queue in T-12 .. T+5 and a queue at T+6; ongoing = >= 2 queued history cells, a link with >= 2, IoU(persistence, T+1 .. T+6 on
P.E) <= 0.9):
  V0 the true state (candidates as is)
  V1 the visible observation (Task 1 blank hidden; queue = finite & speed <= vcut & eligible) for coverage, history queue, onset's empty
     history and persistence (the last history row T-5); the horizon is the true label
  V2 V0 with persistence from the true state at T
  V3 V1 with persistence from the visible state at T
  V4 / V5 = V1 / V3 with the horizon also visible (queue, the any-queue test and the IoU on visible eligible cells)
  V6 V1 with persistence = per link the last visible observation of T-12 .. T-1 (ffill); V7 V1 with the IoU averaged over rows
  V8 V6 with the persistence IoU read on rows T .. T+25; V9 V6 with the horizon IoU on visible eligible cells; V10 V6 with the horizon
     labels and eligibility visible; V11 V6 with the ffill window T-55 .. T (row T included); V12 V11 with coverage / history queue on T-55 .. T; V11d = V11 as deployed by t2_selcond (row T = the true T+5 on links visible at T-5)
  V6E / V11RE (#286) = V6 / V11 (V11 = V11R: the visible cells of row T) with the coverage on the eligible share (pre-mask)
  V11P / V6P (#284) = every condition on the pre-mask layer (finite & eligible), ffill over T-55 .. T / T-60 .. T-5
For each official window: qualifies at T; first origin of its run (T-5 does not qualify); the first qualifying origin at or after the
previous official window of the panel (either condition) + 360 min (the month start for the first). Counts per variant x condition.
usage: py -B -X utf8 g354_selector_id.py"""
import os, sys, pickle
os.environ["TFB_T2_NOT"] = "1"
import numpy as np, pandas as pd
sys.path.insert(0, r"C:/Users/user/tfb/src")
from cachelib import load, T2_PANELS
from t2_windows import queue_matrix, iou, H, F
A = pickle.load(open("C:/Users/user/tfb/work/t2/g342_jun80.pkl", "rb"))
GAP = int(os.environ.get("TFB_SEL_GAP", "72")); TAU = float(os.environ.get("TFB_SEL_TAU", "0.9")); COV = float(os.environ.get("TFB_SEL_COV", "0.7")); SAMEC = os.environ.get("TFB_SEL_SAMECOND", "0") == "1"; VARS_ = os.environ.get("TFB_SEL_VARS", "V0,V1,V2,V3,V4,V5").split(",")   # rows after the previous official origin (72 = 360 min; 78 = 360 min after its forecast end)


def qualify(P, Qm, var):
    Vv = np.where(P.blank, np.nan, P.V); vq = np.isfinite(Vv) & (np.nan_to_num(Vv, nan=1e9) <= P.vcut[None, :]) & P.E
    vis = var in ("V1", "V3", "V4", "V5", "V6", "V7", "V8", "V9", "V10", "V11", "V12", "V11d", "V11P", "V6P", "V6E", "V11RE"); fv = var in ("V4", "V5"); obs = (np.isfinite(Vv) if vis else np.isfinite(P.V)) & P.E; Hq = vq if vis else Qm
    Ft = vq if fv else Qm; any_t = Ft.any(axis=1); any_h = Hq.any(axis=1); cnt = np.concatenate([np.zeros((1, P.n), np.int32), np.cumsum(Hq, axis=0, dtype=np.int32)])
    ocnt = np.concatenate([[0], np.cumsum((P.E if var in ('V6E', 'V11RE') else obs).sum(axis=1))]);   # #286: V6E / V11RE coverage = the eligible share (pre-mask)
    qt = np.concatenate([[0], np.cumsum(any_t)]); qh = np.concatenate([[0], np.cumsum(any_h)])
    on = np.zeros(P.T, bool); og = np.zeros(P.T, bool); jun = np.flatnonzero(np.array([s.startswith("2030-06") for s in P.times]))
    for t in range(max(H, jun[0]), min(P.T - F, jun[-1] + 1)):
        sh = 1 if var == "V12" else 0   # V12: coverage / history queue / onset history on T-55 .. T
        if P.times[t - H][:10] != P.times[t + F][:10] or (ocnt[t + sh] - ocnt[t - H + sh]) / (H * P.n) < COV or not qt[t + F + 1] - qt[t + 1] > 0:
            continue
        if qh[t + sh] - qh[t - H + sh] == 0 and qt[t + F] - qt[t] == 0 and any_t[t + F]:
            on[t] = True
        else:
            hq = cnt[t + sh] - cnt[t - H + sh]
            if hq.sum() >= 2 and hq.max() >= 2:
                if var in ("V11P", "V6P"):   # #284: everything on the pre-mask layer (finite & eligible); V11P ffill over T-55 .. T, V6P over T-60 .. T-5
                    ob_ = np.isfinite(P.V) & P.E; a0, a1 = (t - H + 1, t + 1) if var == "V11P" else (t - H, t)
                    seen = ob_[a0:a1]; last = np.where(seen.any(axis=0), H - 1 - np.argmax(seen[::-1], axis=0), -1); ps = np.where(last >= 0, Qm[a0:a1][np.maximum(last, 0), np.arange(P.n)], False)
                elif var == "V11d":   # #283 as deployed: row T = the true T+5 state on the links visible at T-5, the box (ffill T-60 .. T-5) elsewhere
                    seen = np.isfinite(Vv[t - H:t]) & P.E[t - H:t]; last = np.where(seen.any(axis=0), H - 1 - np.argmax(seen[::-1], axis=0), -1); bx = np.where(last >= 0, vq[t - H:t][np.maximum(last, 0), np.arange(P.n)], False)
                    ps = np.where(np.isfinite(Vv[t - 1]) & P.E[t - 1], Qm[t + 1], bx)
                elif var in ("V6", "V8", "V9", "V10", "V11", "V12", "V6E", "V11RE"):   # per link the last visible observation in T-12 .. T-1 (ffill); V11 also row T
                    a0, a1 = (t - H + 1, t + 1) if var in ("V11", "V12", "V11RE") else (t - H, t)
                    seen = np.isfinite(Vv[a0:a1]) & P.E[a0:a1]; last = np.where(seen.any(axis=0), H - 1 - np.argmax(seen[::-1], axis=0), -1)
                    ps = np.where(last >= 0, vq[a0:a1][np.maximum(last, 0), np.arange(P.n)], False)
                else:
                    ps = {"V0": Qm[t - 1], "V1": vq[t - 1], "V2": Qm[t], "V3": vq[t], "V4": vq[t - 1], "V5": vq[t], "V7": vq[t - 1]}[var]
                fut = Ft[t + 1:t + F + 1]; el = (obs if fv else P.E)[t + 1:t + F + 1]
                if var == "V9":   # horizon IoU on the visible eligible cells, true labels
                    el = obs[t + 1:t + F + 1]
                if var == "V10":   # horizon fully visible (labels and eligibility), the rest as V6
                    fut = vq[t + 1:t + F + 1]; el = obs[t + 1:t + F + 1]
                if var == "V8":   # the horizon one row earlier (T .. T+25)
                    fut = Ft[t:t + F]; el = P.E[t:t + F]
                if var == "V7":   # IoU per row, averaged
                    og[t] = np.mean([iou(ps[None], fut[r:r + 1], el[r:r + 1]) for r in range(F)]) <= TAU
                else:
                    og[t] = iou(np.broadcast_to(ps, fut.shape), fut, el) <= TAU
    return on, og, int(jun[0])


rows = []
for p in T2_PANELS:
    P = load(p, "train"); Qm = queue_matrix(P); rs = sorted([x for x in A["rows"] if x["panel"] == p], key=lambda x: x["t"])
    for var in VARS_:
        on, og, j0 = qualify(P, Qm, var)
        for i, r in enumerate(rs):
            t = r["t"]; q = on if r["cond"] == "queue_onset" else og; pv = [x["t"] for x in rs[:i] if (x["cond"] == r["cond"] or not SAMEC)]
            lo = (max(pv) + GAP) if pv else j0; nx = np.flatnonzero(q[lo:]); first = int(lo + nx[0]) if len(nx) else -1
            rows.append(dict(var=var, cond=r["cond"], qual=bool(q[t]), first_run=bool(q[t] and not q[t - 1]), first_after_prev=first == t, off_rows=(t - first) if first >= 0 else np.nan))
    load.cache_clear(); print(p, "done", flush=True)
D = pd.DataFrame(rows); D.to_csv(f"C:/Users/user/tfb/work/t2/g354_selector_gap{GAP}_tau{TAU}.csv", index=False); print("GAP", GAP, "TAU", TAU, "COV", COV, "SAMECOND", SAMEC, flush=True)
print(D.groupby(["var", "cond"]).agg(qualifies=("qual", "sum"), first_of_run=("first_run", "sum"), first_after_prev_360=("first_after_prev", "sum"), median_rows_after_first=("off_rows", "median"), max_rows_after_first=("off_rows", "max")).to_string(), flush=True)
