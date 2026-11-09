"""Test-time augmentation for the April-window NN (strategy session 10/22; STATE.md pre-registration): the M3 windows'
features rebuilt from K = 8 perturbed copies of the validation month - multiplicative measurement noise of the twin
generator's size on the raw arrays (speed V x (1 + 0.008 e), flow Q x (1 + 0.023 e), e ~ N(0, 1) per cell) and one
uniform demand level per draw (flow x lambda_k, lambda_k evenly spaced over [0.97, 1.03]). Labels, the scoring mask
and the post-processing (evidence, far gate) stay on the unperturbed data; only the NN inputs change.
Same windows, order and feature code as t2_m3_build.py (TFB_T2_NOT=1, TFB_T2_ALLFIN=1, priors zeroed).
Writes NN/nn_test_m3tta<k>/{X,off,window_id,cond}.npy, k = 1..8 (t2_nn_cpu_infer.py input).
k = 0 is the identity check: no perturbation, the features must equal nn_test_m3 exactly (nothing is written).
usage: python t2_m3_tta_build.py [k list, default 1,...,8]"""
import os, sys, copy, time, pickle, warnings, numpy as np
os.environ["TFB_T2_NOT"] = "1"; os.environ["TFB_T2_ALLFIN"] = "1"; warnings.filterwarnings("ignore")
from cachelib import load, T2_PANELS
from common import NN
from t2_model import PanelInputs
from t2_cache import window_X
from t2_arrival import arrival_features, _dmin
from t2_demand import demand_features, profile

K = 8; SIG_V, SIG_Q = 0.008, 0.023; LAM = np.linspace(0.97, 1.03, K)
KS = [int(k) for k in sys.argv[1].split(",")] if len(sys.argv) > 1 else list(range(1, K + 1))
names = [str(x) for x in np.load(f"{NN}/nn_data_afdem/names.npy", allow_pickle=True)]
I_PON, I_POG = names.index("prior_on"), names.index("prior_og")
VER = os.environ.get("TFB_M3VER", "")                         # 10/23: "v3" -> the v3 window set (nn_test_m3v3 -> nn_test_m3v3tta<k>)
META = pickle.load(open(f"C:/Users/user/tfb/work/t2/m3{VER}_meta.pkl", "rb"))["meta"]
WID0 = [str(w) for w in np.load(f"{NN}/nn_test_m3{VER}/window_id.npy", allow_pickle=True)]
assert [m["window_id"] for m in META] == WID0, "m3_meta order differs from nn_test_m3"
OFF0 = np.load(f"{NN}/nn_test_m3{VER}/off.npy")


def perturbed(P, rng, lam):
    Pc = copy.copy(P)                                           # load() is cached: never modify P itself
    Pc.V = P.V * (1 + SIG_V * rng.standard_normal(P.V.shape))
    Pc.Q = P.Q * (1 + SIG_Q * rng.standard_normal(P.Q.shape)) * lam
    return Pc


t0 = time.time()
for k in KS:
    rows = []; rng = np.random.default_rng(1000 + k)
    for p in T2_PANELS:
        ms = [m for m in META if m["panel"] == p]
        if not ms:
            continue
        P0 = load(p, "validation"); P = perturbed(P0, rng, LAM[k - 1]) if k else copy.copy(P0); PI = PanelInputs(P); prof = profile(load(p, "train"))
        for m in ms:
            t, c = m["t"], m["cond"]
            Xw, _ = window_X(PI, t, c); Xw[:, I_PON] = 0.0; Xw[:, I_POG] = 0.0
            V, Q, tl = PI.window(t); Xa, _ = arrival_features(P, V, Q, tl, PI.x)
            Xd = demand_features(P, Q, tl, t, P.times[t][:10], prof, _dmin(P, PI.x))
            rows.append((m["window_id"], c, np.column_stack([Xw, Xa, Xd]).astype(np.float32)))
        load.cache_clear()
    pos = {w: i for i, (w, _, _) in enumerate(rows)}; rows = [rows[pos[w]] for w in WID0]   # nn_test_m3 order
    off = np.concatenate([[0], np.cumsum([len(r[2]) for r in rows])]); assert (off == OFF0).all(), "row counts differ"
    if k == 0:
        X0 = np.load(f"{NN}/nn_test_m3{VER}/X.npy"); X1 = np.vstack([r[2] for r in rows])
        print("identity check: features equal nn_test_m3:", bool(np.array_equal(X0, X1, equal_nan=True)), f"{time.time() - t0:.0f}s"); continue
    d = f"{NN}/nn_test_m3{VER}tta{k}"; os.makedirs(d, exist_ok=True)
    np.save(f"{d}/X.npy", np.vstack([r[2] for r in rows])); np.save(f"{d}/off.npy", off)
    np.save(f"{d}/window_id.npy", np.array(WID0)); np.save(f"{d}/cond.npy", np.array([r[1] for r in rows]))
    print("draw", k, "lambda", round(float(LAM[k - 1]), 4), "windows", len(rows), f"{time.time() - t0:.0f}s", flush=True)
