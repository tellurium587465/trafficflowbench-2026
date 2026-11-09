"""Twin stage 3: NN inputs for the twin worlds with the SAME feature code as the real pipeline (80 columns =
t2_cache.window_X 67 (TFB_T2_ALLFIN=1, TFB_T2_NOT=1) + t2_arrival 7 + t2_demand 6), so a model trained in world A and
scored in the test worlds goes through exactly the code the submissions use.

World A (training): every candidate origin of the t2_windows rules with >= 30 min spacing (training pool, is_sel False)
plus the selector-like windows (spacing 6 h, is_eval/is_sel True, the local month-CV yardstick). Demand profiles
leave the window's month out (as demcache). Test worlds: selector-like windows only, profiles from all of world A.
Writes E:/tfb_nn/nn_data_twinA (train layout) and E:/tfb_nn/nn_test_twinT (test layout, window ids
'TW_<panel>_<world>_<k>'), plus E:/tfb_nn/twin_labels.pkl {window_id: (label, elig, panel, world, t, cond, date)}.
"""
import os, sys, time, pickle, numpy as np
os.environ["TFB_T2_NOT"] = "1"; os.environ["TFB_T2_ALLFIN"] = "1"
sys.path.insert(0, r"C:/Users/user/tfb/src")
from cachelib import load, T2_PANELS
from t2_model import PanelInputs
from t2_cache import window_X
from t2_arrival import arrival_features, _dmin
from t2_demand import demand_features, profile
from t2_windows import candidates
from common import NN                                        # 10/22: data root off the failing E: (TFB_NN)

# build_nn2: python build_nn2.py <base world, e.g. A6> <extra train family or -> <n extra> <test family> <n test> <suffix>
BASE, RFAM, NR, TFAM, NT, SUFX = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4], int(sys.argv[5]), sys.argv[6]
t0 = time.time()


def windows(P, spacing, seed):
    sel, Qm = candidates(P, np.random.default_rng(seed), spacing)
    return [(c, t) for c, ts in sel.items() for t in ts], Qm


def feats(PI, PT, t, cond, prof):
    X1, n1 = window_X(PI, t, cond)
    V, Q, tl = PI.window(t)
    A, na = arrival_features(PI.P, V, Q, tl, PI.x)
    Dm = demand_features(PI.P, Q, tl, t, str(PI.P.times[t])[:10], prof, _dmin(PI.P, PI.x))
    return np.column_stack([X1, A, Dm]).astype(np.float32), n1 + [f"arr{i}" for i in range(len(na))] + [f"dem{h}_c" for h in (10, 20, 30)] + ["dem30_over_dnmin", "dem_shape30", "dem_level"]


def collect(P, PI, PT, wins, flags, prof_of, Qm, rows, meta, labels, wid_prefix=None):
    names = None                                                    # a world can have no window at all (no queue)
    for (c, t), fl in zip(wins, flags):
        X, names = feats(PI, PT, t, c, prof_of(t))
        rows.append(X)
        y = Qm[t + 1:t + 7].astype(np.uint8); el = P.E[t + 1:t + 7].astype(np.uint8)
        meta.append(dict(panel=P.panel, t=t, cond=c, date=str(P.times[t])[:10], **fl))
        labels.append((y.T, el.T))
        if wid_prefix is not None:
            meta[-1]["window_id"] = f"{wid_prefix}_{len(meta)}"
    return names


names = None
# ---------------- world A ----------------
rows, meta, labels = [], [], []
for p in T2_PANELS:
    P = load(p, f"twin{BASE}"); P.split = "train"; PI = PanelInputs(P)
    if os.environ.get("BASE_PROFILE_ONLY", "0") == "1":     # 10/22: world A only for the demand profile (build_simR keeps
        pool, sel, Qm = [], [], None                         # only the synthetic-year windows, so A's windows would be dropped)
    else:
        pool, Qm = windows(P, 6, 1); sel, _ = windows(P, 72, 0)
    sels = set(sel); pool = [w for w in pool if w not in sels]
    prof_cache = {}
    def prof_of(t, P=P):
        m = str(P.times[t])[:7]
        if m not in prof_cache:
            prof_cache[m] = profile(P, m)
        return prof_cache[m]
    names = collect(P, PI, P, pool, [dict(is_eval=False, is_sel=False)] * len(pool), prof_of, Qm, rows, meta, labels) or names
    collect(P, PI, P, sel, [dict(is_eval=True, is_sel=True)] * len(sel), prof_of, Qm, rows, meta, labels)
    print(p, "world A windows", len(pool), "+", len(sel), f"{time.time()-t0:.0f}s", flush=True)
    profA = profile(P, None)
    for k in range(NR):
        PR = load(p, f"twin{RFAM}{k}"); PR.split = "train"; PIR = PanelInputs(PR)
        poolR, QmR = windows(PR, int(os.environ.get("TWIN_RSPACING", "6")), 1)   # 10/13: thin the extra worlds (memory)
        names = collect(PR, PIR, P, poolR, [dict(is_eval=False, is_sel=False)] * len(poolR), lambda t: profA, QmR, rows, meta, labels) or names
    if NR:
        print(p, "randomized worlds", NR, "total windows", len(meta), f"{time.time()-t0:.0f}s", flush=True)
d = f"{NN}/nn_data_{SUFX}"; os.makedirs(d, exist_ok=True)
off = np.concatenate([[0], np.cumsum([len(r) for r in rows])])
X = np.lib.format.open_memmap(f"{d}/X.npy", mode="w+", dtype=np.float32, shape=(int(off[-1]), int(rows[0].shape[1])))
for k, r in enumerate(rows):
    X[off[k]:off[k + 1]] = r
X.flush(); del X
real = [str(x) for x in np.load(f"{NN}/nn_data_afdem/names.npy", allow_pickle=True)]
assert [str(x) for x in names] == real, ("column names differ from the real NN data", [a for a, b in zip(names, real) if a != b][:5])
np.save(f"{d}/off.npy", off); np.save(f"{d}/names.npy", np.array(names))
for f in ("panel", "t", "cond", "date", "is_eval", "is_sel"):
    np.save(f"{d}/{f}.npy", np.array([m[f] for m in meta]))
np.save(f"{d}/Y.npy", np.vstack([y for y, _ in labels])); np.save(f"{d}/EL.npy", np.vstack([e for _, e in labels]))
print(SUFX, "done", len(meta), "windows", off[-1], "rows", f"{time.time()-t0:.0f}s", flush=True)

# ---------------- test worlds ----------------
if NT == 0:                                                         # 10/22: nothing to write (rows[0] would fail)
    sys.exit(0)
rows, meta, labels = [], [], []; LAB = {}
for p in T2_PANELS:
    PA = load(p, f"twin{BASE}"); PA.split = "train"; profA = profile(PA, None)
    for k in range(NT):
        P = load(p, f"twin{TFAM}{k}"); P.split = "train"; PI = PanelInputs(P)
        sel, Qm = windows(P, 72, 0)
        n0 = len(meta)
        collect(P, PI, PA, sel, [dict(is_eval=True, is_sel=True, world=f"{TFAM}{k}")] * len(sel), lambda t: profA, Qm, rows, meta, labels,
                wid_prefix=f"TW_{p}_{TFAM}{k}")
    print(p, "test worlds", len(meta), f"{time.time()-t0:.0f}s", flush=True)
d = f"{NN}/nn_test_{SUFX}"; os.makedirs(d, exist_ok=True)
off = np.concatenate([[0], np.cumsum([len(r) for r in rows])])
X = np.lib.format.open_memmap(f"{d}/X.npy", mode="w+", dtype=np.float32, shape=(int(off[-1]), int(rows[0].shape[1])))
for k, r in enumerate(rows):
    X[off[k]:off[k + 1]] = r
X.flush(); del X
np.save(f"{d}/off.npy", off); np.save(f"{d}/window_id.npy", np.array([m["window_id"] for m in meta]))
np.save(f"{d}/cond.npy", np.array([m["cond"] for m in meta]))
pickle.dump({m["window_id"]: (y, e, m["panel"], m["world"], m["t"], m["cond"], m["date"]) for m, (y, e) in zip(meta, labels)},
            open(f"{NN}/twin_labels_{SUFX}.pkl", "wb"))
print("test worlds done", len(meta), "windows", f"{time.time()-t0:.0f}s", flush=True)
