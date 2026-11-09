"""TTA inputs for the OFFICIAL Task 2 windows (10/22; the part adopted on M3, t2_m3_tta_build.py): the features of every
validation / private window rebuilt from K = 8 perturbed copies of the month (speed x (1 + 0.008 e), flow x (1 + 0.023 e)
x lambda_k, lambda_k evenly spaced over [0.97, 1.03]; e ~ N(0, 1) per cell of the fixed calendar grid, so a release cut
after T gives the same noise on the rows <= T). Same features as t2_nn_af.py (nn_test_afdem): window_X + arrival +
demand, priors zeroed, TFB_T2_NOT=1, TFB_T2_ALLFIN=1. k = 0 is the identity check against nn_test_afdem (nothing written).
Writes NN/nn_test_afdem_tta<k>/{X,off,window_id,cond}.npy for t2_nn_cpu_infer.py.
usage: python t2_tta_official_build.py [k list, default 1,...,8]"""
import os, sys, copy, time, shutil, warnings, numpy as np, pandas as pd
os.environ["TFB_T2_NOT"] = "1"; os.environ["TFB_T2_ALLFIN"] = "1"; warnings.filterwarnings("ignore")
from cachelib import load, T2_PANELS
from common import REL, NN
from t2_model import PanelInputs
from t2_cache import window_X
from t2_arrival import arrival_features, _dmin
from t2_demand import demand_features, profile

K = 8; SIG_V, SIG_Q = 0.008, 0.023; LAM = np.linspace(0.97, 1.03, K)
ORDER = [(sp, p) for sp in ("validation", "private") for p in T2_PANELS]   # the noise stream of draw k runs through these in order


def perturb(P0, rng, k):
    """the month P0 as draw k sees it (k = 0: as released); consumes V.shape then Q.shape normals from rng"""
    P = copy.copy(P0)
    if k:
        P.V = P0.V * (1 + SIG_V * rng.standard_normal(P0.V.shape)); P.Q = P0.Q * (1 + SIG_Q * rng.standard_normal(P0.Q.shape)) * LAM[k - 1]
    return P


def skip(rng, shapes):
    """10/23 (t2_window_entry.py): consume the draws of the (split, panel) months before the window's, given their shapes"""
    for sv, sq in shapes:
        rng.standard_normal(tuple(sv)); rng.standard_normal(tuple(sq))


if __name__ == "__main__":
    KS = [int(k) for k in sys.argv[1].split(",")] if len(sys.argv) > 1 else list(range(1, K + 1))
    names = [str(x) for x in np.load(f"{NN}/nn_data_afdem/names.npy", allow_pickle=True)]
    I_PON, I_POG = names.index("prior_on"), names.index("prior_og")
    SRC = f"{NN}/nn_test_afdem"; WID0 = [str(w) for w in np.load(f"{SRC}/window_id.npy", allow_pickle=True)]; OFF0 = np.load(f"{SRC}/off.npy")
    t0 = time.time()
    for k in KS:
        rows, ids = [], []; rng = np.random.default_rng(2000 + k)
        for split, p in ORDER:
            P0 = load(p, split); P = perturb(P0, rng, k)
            PI = PanelInputs(P); ti = {s: i for i, s in enumerate(P.times)}; prof = profile(load(p, "train"))
            for r in pd.read_csv(REL / "task2" / p / split / "window_index.csv").itertuples():
                t = ti[r.forecast_origin]
                Xw, _ = window_X(PI, t, r.condition); Xw[:, I_PON] = 0.0; Xw[:, I_POG] = 0.0
                V, Q, tl = PI.window(t); Xa, _ = arrival_features(P, V, Q, tl, PI.x)
                Xd = demand_features(P, Q, tl, t, P.times[t][:10], prof, _dmin(P, PI.x))
                rows.append(np.column_stack([Xw, Xa, Xd]).astype(np.float32)); ids.append(r.window_id)
            load.cache_clear()
        assert ids == WID0, "window order differs from nn_test_afdem"
        X = np.vstack(rows); assert (np.concatenate([[0], np.cumsum([len(r) for r in rows])]) == OFF0).all(), "row counts differ"
        if k == 0:
            print("identity check: features equal nn_test_afdem:", bool(np.array_equal(X, np.load(f"{SRC}/X.npy"), equal_nan=True)), f"{time.time() - t0:.0f}s"); continue
        d = f"{NN}/nn_test_afdem_tta{k}"; os.makedirs(d, exist_ok=True); np.save(f"{d}/X.npy", X)
        for f in ("off.npy", "window_id.npy", "cond.npy"):
            shutil.copyfile(f"{SRC}/{f}", f"{d}/{f}")
        print("draw", k, "lambda", round(float(LAM[k - 1]), 4), "windows", len(ids), f"{time.time() - t0:.0f}s", flush=True)
