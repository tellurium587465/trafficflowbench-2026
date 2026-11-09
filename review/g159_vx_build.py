"""G159 (11/01, ledger #117b): M3 dev network inputs with the speed columns put back on the training scale. The inputs are V / vf; on a redrawn month a link's free-flow level sits at
x = vff(earlier days of the same month, rows <= T) / vff(training months), so each V / vf column of link j is divided by x_j (x = 1 with < 50 free cells, as G155). Flags and
counts that come from the threshold are left as they are (#117: the threshold does not move). Writes two test folders with the dev windows only: nn_test_m3v3dv (inputs unchanged, the
paired base for the same CPU path) and nn_test_m3v3vx (corrected); the hold-out days are not written.
usage: TFB_M3VER=v3 py -B -X utf8 g159_vx_build.py"""
import os, sys, pickle, warnings
os.environ.setdefault("TFB_M3VER", "v3"); sys.dont_write_bytecode = True; sys.path.insert(0, r"C:/Users/user/tfb/src"); sys.argv = [sys.argv[0], "0,1,2,3"]
import numpy as np
import t2_m3_gate as G, m3v2
from cachelib import load
from common import NN

W = G.W; VFF = pickle.load(open(W + "g120_panel_stats.pkl", "rb"))[0]
names = [str(x) for x in np.load(f"{NN}/nn_data_afdem/names.npy", allow_pickle=True)]; S0 = f"{NN}/nn_test_m3v3"; X = np.load(S0 + "/X.npy", mmap_mode="r"); off = np.load(S0 + "/off.npy"); wid = [str(w) for w in np.load(S0 + "/window_id.npy", allow_pickle=True)]; cond = np.load(S0 + "/cond.npy", allow_pickle=True)
assert X.shape[1] == len(names), (X.shape, len(names))
VC = [names.index(c) for c in ("v_T", "v_T1", "v_T2", "v_m3", "v_m12", "v_min", "last_v", "v_trend")]; NB = [(names.index(f"nb{o}_v"), o) for o in (-5, -3, -2, -1, 1, 2, 3, 5)]
ASOF = {}


def x_of(p, day):
    if (p, day) not in ASOF:
        P = load(p, "validation"); rr = slice(0, day * 288); V_ = P.V[rr]; ff = np.where(P.E[rr] & (V_ >= 0.85 * P.vf[None, :]), V_, np.nan); n = np.isfinite(ff).sum(axis=0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore"); v = np.nanmedian(ff, axis=0) if len(ff) else np.full(P.n, np.nan)
        ASOF[(p, day)] = np.where((n >= 50) & np.isfinite(VFF[p]), v / VFF[p], 1.0)
    return ASOF[(p, day)]


keep = [i for i, w in enumerate(wid) if not m3v2.is_holdout(np.array([G.META[w]["day"]]))[0]]; n0 = np.array([off[i + 1] - off[i] for i in keep]); of2 = np.r_[0, np.cumsum(n0)]
A = np.concatenate([np.array(X[off[i]:off[i + 1]]) for i in keep]); B = A.copy(); sx = []
for r, i in enumerate(keep):
    m = G.META[wid[i]]; a, b = of2[r], of2[r + 1]; x = x_of(m["panel"], m["day"]).astype(np.float32); assert b - a == len(x); sx.append(float(np.std(np.log(x))))
    B[a:b, VC] = A[a:b, VC] / x[:, None]
    for c, o in NB:
        B[a:b, c] = A[a:b, c] / x[np.clip(np.arange(len(x)) + o, 0, len(x) - 1)]
for nm, D in (("nn_test_m3v3dv", A), ("nn_test_m3v3vx", B)):
    d = f"{NN}/{nm}"; os.makedirs(d, exist_ok=True); np.save(d + "/X.npy", D); np.save(d + "/off.npy", of2); np.save(d + "/window_id.npy", np.array([wid[i] for i in keep], dtype=object)); np.save(d + "/cond.npy", cond[keep])
print(f"windows {len(keep)} | onset {int((cond[keep] == 'queue_onset').sum())} ongoing {int((cond[keep] == 'queue_ongoing').sum())} | sd of log x per window: mean {np.mean(sx):.4f} | columns {X.shape[1]} | cells changed {float(np.mean(A[np.isfinite(A)] != B[np.isfinite(A)])):.3f}")
