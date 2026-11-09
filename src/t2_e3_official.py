"""E3' (ongoing) and the onset shift on the OFFICIAL windows (10/23; the M3 versions are t2_m3v2_prev_build.py /
t2_m3v2_e3.py). For each official window on 0-based day d of its month, the pseudo-origins T' = T - b*288, b = 1..min(6, d),
are rebuilt from the released month (the same production features as nn_test_afdem: TFB_T2_NOT=1, TFB_T2_ALLFIN=1, priors
zeroed, demand profile from the train months); their outcome rows T'+1..T'+6 lie on completed earlier days (the release's
own blanks are already missing). The onset model (s46 spec's family q24f, CFG5 weights, seeds 0-3) predicts them; per
link and step, kq = queued and observed, ko = observed, pbar = the mean prediction where observed; then
logit p' = logit p + beta (logit pi - logit pbar) + c with pi = (kq + a pbar) / (ko + a) where observed, as t2_m3v2_e3.
Constants chosen on the M3 v3 windows outside the hold-out (10/23 05:40): onset (a, beta, c) = (1, 0, 0.5),
ongoing (1, 1, 0).
usage: python t2_e3_official.py build        (features -> NN/nn_test_offprev6, work/t2/offprev6_meta.pkl, CPU inference)
       import t2_e3_official as E3; E3.apply(probs)   (probs: {window_id: (6, n)} -> corrected copy)"""
import os, sys, pickle, time, json, subprocess
import numpy as np

SRC = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, SRC)
from common import WORK
W = WORK + "/t2/"
CONST = {"queue_onset": (1.0, 0.0, 0.5), "queue_ongoing": (1.0, 1.0, 0.0)}
CFG5 = lambda f: [(f + "_attnglob", 2), (f + "_attnwp05", 2), (f + "_attn1", 2), (f + "_drop1", 1), (f + "_r13", 1)]

if __name__ == "__main__" and sys.argv[1:] == ["build"]:
    os.environ["TFB_T2_NOT"] = "1"; os.environ["TFB_T2_ALLFIN"] = "1"
    import pandas as pd
    from cachelib import load, T2_PANELS
    from common import NN, REL
    from t2_model import PanelInputs
    from t2_cache import window_X
    from t2_arrival import arrival_features, _dmin
    from t2_demand import demand_features, profile
    from t2_windows import queue_matrix
    names = [str(x) for x in np.load(f"{NN}/nn_data_afdem/names.npy", allow_pickle=True)]
    I_PON, I_POG = names.index("prior_on"), names.index("prior_og")
    t0 = time.time(); rows, wid, cond, meta = [], [], [], []
    for sp in ("validation", "private"):
        for p in T2_PANELS:
            P = load(p, sp); PI = PanelInputs(P); prof = profile(load(p, "train")); ti = {s: i for i, s in enumerate(P.times)}
            Qm = queue_matrix(P); obs = np.isfinite(P.V) & P.E
            for r in pd.read_csv(REL / "task2" / p / sp / "window_index.csv").itertuples():
                t = ti[r.forecast_origin]; d = t // 288
                for b in range(1, min(6, d) + 1):
                    tp = t - b * 288
                    if tp < 12:
                        break
                    Xw, _ = window_X(PI, tp, r.condition); Xw[:, I_PON] = 0.0; Xw[:, I_POG] = 0.0
                    V, Q, tl = PI.window(tp); Xa, _ = arrival_features(P, V, Q, tl, PI.x)
                    Xd = demand_features(P, Q, tl, tp, P.times[tp][:10], prof, _dmin(P, PI.x))
                    rows.append(np.column_stack([Xw, Xa, Xd]).astype(np.float32)); wid.append(f"{r.window_id}_b{b}"); cond.append(r.condition)
                    mo = obs[tp + 1:tp + 7]
                    meta.append(dict(window_id=f"{r.window_id}_b{b}", parent=r.window_id, b=b, tp=tp, y=(Qm[tp + 1:tp + 7] & mo).T.copy(), m=mo.T.copy()))
            load.cache_clear()
    d_ = f"{NN}/nn_test_offprev6"; os.makedirs(d_, exist_ok=True)
    np.save(f"{d_}/X.npy", np.vstack(rows)); np.save(f"{d_}/off.npy", np.concatenate([[0], np.cumsum([len(x) for x in rows])]))
    np.save(f"{d_}/window_id.npy", np.array(wid)); np.save(f"{d_}/cond.npy", np.array(cond))
    pickle.dump(meta, open(W + "offprev6_meta.pkl", "wb")); print("official pseudo-origins", len(meta), f"{time.time() - t0:.0f}s", flush=True)
    tags = ",".join(f"{f}_s{s}" for s in range(4) for f, _ in CFG5("q24f"))
    subprocess.run([sys.executable, "t2_nn_cpu_infer.py", tags, "nn_test_offprev6", "offprev6"], cwd=SRC, check=True)
    sys.exit()

_S = {}
def _stats():
    if not _S:
        meta = pickle.load(open(W + "offprev6_meta.pkl", "rb"))
        parts = [(w, pickle.load(open(W + f"probs_offprev6__{f}_s{s}.pkl", "rb"))) for s in range(4) for f, w in CFG5("q24f")]
        tw = sum(w for w, _ in parts)
        for e in meta:
            pp = sum(w * pr[e["window_id"]] for w, pr in parts) / tw            # (6, n)
            kq, ko, sp = _S.get(e["parent"], (0.0, 0.0, 0.0))
            mo = e["m"].T.astype(float); _S[e["parent"]] = (kq + e["y"].T.astype(float), ko + mo, sp + pp * mo)
    return _S


def correct(p, kq, ko, sp, a, beta, c):
    """one window: logit p' = logit p + beta (logit pi - logit pbar) + c where the earlier days observed the cell"""
    lg = lambda x: np.log(x / (1 - x)); clip = lambda x: np.clip(x, 1e-4, 1 - 1e-4); has = ko > 0
    pbar = clip(np.where(has, sp / np.maximum(ko, 1), 0.5)); pi = clip((kq + a * pbar) / (ko + a))
    q = 1 / (1 + np.exp(-(lg(clip(p)) + np.where(has, beta * (lg(pi) - lg(pbar)) + c, 0.0))))
    return np.where(p > 0, q, 0.0)


def apply(probs, cond):
    """probs {window_id: (6, n)}, cond {window_id: condition} -> corrected copy (windows without earlier days unchanged)"""
    S = _stats()
    return {w: (correct(p, *S[w], *CONST[cond[w]]) if w in S else p) for w, p in probs.items()}


def earlier_days(P, t):
    """10/23 (t2_window_entry.py): (kq, ko) of the window at row t of the month P, as 'build' collects them (b = 1 ..
    min(6, day), outcome rows T'+1..T'+6 on completed earlier days), or None without earlier days. pbar needs the
    pseudo-origins' predictions; the onset shift (beta = 0) does not use it."""
    from t2_windows import queue_matrix
    Qm = queue_matrix(P); obs = np.isfinite(P.V) & P.E; kq = ko = None
    for b in range(1, min(6, t // 288) + 1):
        tp = t - b * 288
        if tp < 12:
            break
        mo = obs[tp + 1:tp + 7]; y = (Qm[tp + 1:tp + 7] & mo).astype(float)
        kq, ko = (y, mo.astype(float)) if kq is None else (kq + y, ko + mo.astype(float))
    return None if kq is None else (kq, ko)
