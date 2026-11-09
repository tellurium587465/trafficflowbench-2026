"""The organisers' Task 2 check, run locally (discussion 749601, ~10/19): Task 2 is re-run with every row after each
window's forecast origin T deleted, and the predictions must not change.
Per validation/private window, the NN input features are rebuilt exactly as t2_nn_af.py builds nn_test_afdem
(TFB_T2_NOT=1, TFB_T2_ALLFIN=1: window_X + arrival + demand columns, priors zeroed), twice:
  full  : from the released month as stored
  trunc : from a copy of the month whose observation rows >= T are deleted (every time-indexed float array -> NaN,
          bool -> False; rows >= T includes the origin row itself, the stricter reading)
and both are compared with the stored nn_test_afdem rows. Identical features give identical network outputs. The two
post-processing steps read rows < T only (t2_onset_evidence: slots on earlier days; t2_far_gate: the same day 05:00..T-1
and slots on earlier days), checked in the code on 10/22.
usage: python t2_causality_check.py"""
import os, copy, warnings, numpy as np, pandas as pd
os.environ["TFB_T2_NOT"] = "1"; os.environ["TFB_T2_ALLFIN"] = "1"; warnings.filterwarnings("ignore")
from cachelib import load, T2_PANELS
from common import REL, NN
from t2_model import PanelInputs
from t2_cache import window_X
from t2_arrival import arrival_features, _dmin
from t2_demand import demand_features, profile

names = [str(x) for x in np.load(f"{NN}/nn_data_afdem/names.npy", allow_pickle=True)]
I_PON, I_POG = names.index("prior_on"), names.index("prior_og")
SX = np.load(f"{NN}/nn_test_afdem/X.npy", mmap_mode="r"); SO = np.load(f"{NN}/nn_test_afdem/off.npy")
SW = [str(w) for w in np.load(f"{NN}/nn_test_afdem/window_id.npy", allow_pickle=True)]; pos = {w: i for i, w in enumerate(SW)}


def feats(P, PI, t, cond, prof):
    Xw, _ = window_X(PI, t, cond); Xw[:, I_PON] = 0.0; Xw[:, I_POG] = 0.0
    V, Q, tl = PI.window(t); Xa, _ = arrival_features(P, V, Q, tl, PI.x)
    Xd = demand_features(P, Q, tl, t, P.times[t][:10], prof, _dmin(P, PI.x))
    return np.column_stack([Xw, Xa, Xd]).astype(np.float32)


def truncated(P, t):
    Pc = copy.copy(P)
    for k in sorted(P._lazy):
        a = getattr(P, k)                                     # forces the lazy load on the original
        if isinstance(a, np.ndarray) and a.ndim >= 1 and a.shape[0] == P.T and k not in ("times",):
            b = a.copy()
            if b.dtype.kind == "f":
                b[t:] = np.nan
            elif b.dtype.kind == "b":
                b[t:] = False
            else:
                continue                                      # integer masks / regimes are not observations
            setattr(Pc, k, b)
    return Pc


def same(a, b):
    return a.shape == b.shape and np.array_equal(a, b, equal_nan=True)


rows = []
for split in ("validation", "private"):
    for p in T2_PANELS:
        P = load(p, split); PI = PanelInputs(P); ti = {s: i for i, s in enumerate(P.times)}; prof = profile(load(p, "train"))
        for r in pd.read_csv(REL / "task2" / p / split / "window_index.csv").itertuples():
            t = ti[r.forecast_origin]; Ff = feats(P, PI, t, r.condition, prof)
            P2 = truncated(P, t); Ft = feats(P2, PanelInputs(P2), t, r.condition, prof)
            i = pos[r.window_id]; Fs = np.asarray(SX[SO[i]:SO[i + 1]])
            d = np.nanmax(np.abs(np.nan_to_num(Ft, nan=1e9) - np.nan_to_num(Ff, nan=1e9))) if Ft.shape == Ff.shape else np.inf
            rows.append(dict(window=r.window_id, cond=r.condition, full_eq_trunc=same(Ff, Ft), full_eq_stored=same(Ff, Fs),
                             max_abs_diff=float(d), cols_differing=int((~np.isclose(Ff, Ft, equal_nan=True)).any(axis=0).sum()) if Ft.shape == Ff.shape else -1))
        print(split, p, "windows", sum(1 for x in rows if x["window"].startswith(f"Q2_{p}_{split}")),
              "trunc-equal", sum(1 for x in rows if x["window"].startswith(f"Q2_{p}_{split}") and x["full_eq_trunc"]), flush=True)
        load.cache_clear()
D = pd.DataFrame(rows); D.to_csv(r"C:/Users/user/tfb/work/t2_causality_check.csv", index=False)
print(f"windows {len(D)}: features unchanged when rows >= T are deleted: {int(D.full_eq_trunc.sum())}; "
      f"rebuilt == stored nn_test_afdem: {int(D.full_eq_stored.sum())}")
if not D.full_eq_trunc.all():
    print(D[~D.full_eq_trunc].head(10).to_string(index=False))
