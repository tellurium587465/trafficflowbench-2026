"""The no-memory LightGBM (t2_lgb_nomem_final.py) applied from its saved model files to the official windows, without the
3 GB training cache (10/23 repro: t2_final_spec.lgb_ongoing / lgb_onset, column names from repro/featcache_names.json).
Same output as `t2_lgb_nomem_final.py apply` (checked bit-identical on the 160 windows, 10/23).
usage: python t2_lgbnm_apply.py [out=lgbnm]   -> work/t2/probs_<out>.pkl"""
import sys, pickle, numpy as np, lightgbm as lgb
import t2_final_spec as FS
from common import NN, WORK

out = sys.argv[1] if len(sys.argv) > 1 else "lgbnm"; d = f"{NN}/nn_test_afdem"
X = np.load(f"{d}/X.npy"); off = np.load(f"{d}/off.npy"); wid = [str(w) for w in np.load(f"{d}/window_id.npy", allow_pickle=True)]
cond = np.load(f"{d}/cond.npy", allow_pickle=True)
M = {c: lgb.Booster(model_file=f"{WORK}/t2/lgbnm_{c}.txt") for c in ("queue_onset", "queue_ongoing")}
res = {w: (FS.lgb_onset if cond[i] == "queue_onset" else FS.lgb_ongoing)(X[off[i]:off[i + 1]], M[str(cond[i])]) for i, w in enumerate(wid)}
pickle.dump(res, open(f"{WORK}/t2/probs_{out}.pkl", "wb")); print(d, len(res), "windows ->", out)
