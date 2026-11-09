"""No-memory LightGBM, FINAL (strategy bet 2 (i), 10/22): the 10/10 configuration of l127_full_af_nomem (cache 2_noT_af +
arrival + demand, location / prior / static / time-of-day / weekday columns dropped, num_leaves 127, rounds onset 600 /
ongoing 1500) trained on all non-selector train windows (as t2_final.py), then applied to
  - the M3 pseudo windows (NN/nn_test_m3: the same 80 columns as the cache, checked equal on the training rows)
  - the official windows (NN/nn_test_afdem)
Writes work/t2/probs_m3__lgbnm.pkl and work/t2/probs_lgbnm.pkl ({window_id: (6, n)}; onset: only the last step).
Needs ~14 GB free RAM: run with the GPU daemon stopped (10/7 hard reset).
usage: python t2_lgb_nomem_final.py              (train + apply)
       python t2_lgb_nomem_final.py apply [out]  (10/23 repro: the saved models -> work/t2/probs_<out>.pkl, default lgbnm)"""
import os, sys, time, pickle
os.environ.update(TFB_T2_NOT="1", TFB_T2CACHE="2_noT_af", TFB_T2_ARR="1", TFB_T2_DEM="1",
                  TFB_T2_DROP="prior_on,prior_og,pos,vf,capl,lanes,length,w,tod,dow")
import numpy as np, lightgbm as lgb
import t2_fast as T
from t2_model import PARAMS
from t2_windows import F
from common import NN, WORK

t0 = time.time(); W = WORK + "/t2/"
APPLY = sys.argv[1:2] == ["apply"]; OUT = sys.argv[2] if APPLY and len(sys.argv) > 2 else "lgbnm"   # 10/23: apply <out name>
prm = dict(PARAMS); prm.update(num_leaves=127, num_threads=14)
pr = T.priors(set())
tr = np.flatnonzero(~T.META["is_sel"])
TMP = os.environ.get("TFB_TMP_LGB", NN + "/../tmp_lgbnm_ongoing.f32")


def design_stream(idx, cond, neg_keep=0.15):
    """T.design(idx, pr, cond) for the ongoing condition, rows streamed to a raw float32 file instead of a list + vstack
    (10/22: the in-memory version peaked at ~16 GB and took free RAM to 0.5 GB). Same rng (default_rng(5)), same window
    order, same keep masks and weights -> the same training rows."""
    rng = np.random.default_rng(5); ys, ws, n = [], [], 0
    with open(TMP, "wb") as f:
        for k in idx:
            if T.META["cond"][k] != cond:
                continue
            X = T.win_X(k, pr); lab, el = T.LAB[k]
            ww = 100.0 / max((lab & el).sum(), 3) ** T.WPOW
            near = (np.minimum(X[:, T.I_DU], X[:, T.I_DD]) <= 6.0) | lab.any(axis=0)
            keep = near | (rng.random(len(X)) < neg_keep); wk = np.where(near, 1.0, 1.0 / neg_keep)
            for s in range(F):
                m = keep & el[s]; Bk = T.step_block(X, s + 1)[m].astype(np.float32)
                f.write(np.ascontiguousarray(Bk).tobytes()); n += len(Bk); ncol = Bk.shape[1]
                ys.append(lab[s][m]); ws.append(ww * wk[m])
    return RawRows(TMP, n, ncol), np.concatenate(ys).astype(np.float32), np.concatenate(ws)


class RawRows(lgb.Sequence):
    """the streamed rows for lgb.Dataset without mapping the whole file (23:00: a memmap of the 7.2 GB file took the
    free RAM to 0.6 GB during Dataset construction): batches are read with np.fromfile; the bin-boundary sample (a list
    of row indices) is read row by row through a short-lived memmap."""
    batch_size = 1 << 16

    def __init__(self, path, n, ncol):
        self.path, self.n, self.ncol = path, n, ncol

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            a, b, _ = idx.indices(self.n)
            with open(self.path, "rb") as f:
                f.seek(a * self.ncol * 4); return np.fromfile(f, dtype=np.float32, count=(b - a) * self.ncol).reshape(b - a, self.ncol).astype(np.float64)
        mm = np.memmap(self.path, dtype=np.float32, mode="r", shape=(self.n, self.ncol))
        out = np.array(mm[idx], dtype=np.float64); del mm               # lgb.Sequence wants float64 (values are exact)
        return out


models = {}
if APPLY:                                                       # 10/23 repro: the saved models only
    models = {cond: lgb.Booster(model_file=W + f"lgbnm_{cond}.txt") for cond in ("queue_onset", "queue_ongoing")}
for cond, nr in (() if APPLY else (("queue_onset", 600), ("queue_ongoing", 1500))):
    X, y, w = T.design(tr, pr, cond) if cond == "queue_onset" else design_stream(tr, cond)
    models[cond] = lgb.train(prm, lgb.Dataset([X] if isinstance(X, lgb.Sequence) else X, y, weight=w, free_raw_data=True), nr); del X, y, w
    models[cond].save_model(W + f"lgbnm_{cond}.txt"); print(cond, "trained", f"{time.time() - t0:.0f}s", flush=True)
if not APPLY:
    os.remove(TMP)


def apply(test_dir, out):
    X = np.load(f"{NN}/{test_dir}/X.npy"); off = np.load(f"{NN}/{test_dir}/off.npy")
    wid = [str(x) for x in np.load(f"{NN}/{test_dir}/window_id.npy", allow_pickle=True)]; cond = np.load(f"{NN}/{test_dir}/cond.npy", allow_pickle=True)
    res = {}
    for i, w in enumerate(wid):
        Xw = X[off[i]:off[i + 1]].copy(); Xw[:, T.I_PON] = 0.0; Xw[:, T.I_POG] = 0.0; Xw = T.apply_drop(Xw)   # priors dropped anyway
        if cond[i] == "queue_onset":
            p = np.zeros((F, len(Xw))); p[-1] = models["queue_onset"].predict(Xw, num_threads=14)
        else:
            p = models["queue_ongoing"].predict(np.vstack([T.step_block(Xw, s + 1) for s in range(F)]), num_threads=14).reshape(F, len(Xw))
        res[w] = p.astype(np.float32)
    pickle.dump(res, open(W + f"probs_{out}.pkl", "wb")); print(test_dir, len(res), "windows ->", out, f"{time.time() - t0:.0f}s", flush=True)


if not APPLY:
    apply("nn_test_m3", "m3__lgbnm")
apply("nn_test_afdem", OUT)
