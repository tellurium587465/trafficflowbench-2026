"""G167 (11/01, ledger #118): the arrays of the Task 1 image network for a private Kaggle dataset. Per panel and split: V and Q in float32, the finite and target masks, E, the statics, the month of each day.
Written to C:/tfb_data/kaggle_up/d_t1img/ (C: only). No label of any test month exists in these arrays: test cells under the mask are NaN in the release.
usage: py -B -X utf8 g167_unet_data.py"""
import os, sys, json, time
sys.dont_write_bytecode = True; sys.path.insert(0, r"C:/Users/user/tfb/src")
import numpy as np
from cachelib import load
from common import PANELS

O = "C:/tfb_data/kaggle_up/d_t1img/"; os.makedirs(O, exist_ok=True); t0 = time.time()
for p in PANELS:
    S = {}
    for sp in ("train", "validation", "private"):
        P = load(p, sp); S[sp + "_V"] = P.V.astype(np.float32); S[sp + "_Q"] = P.Q.astype(np.float32); S[sp + "_fin"] = np.isfinite(P.V) & np.isfinite(P.Q); S[sp + "_target"] = P.target; S[sp + "_E"] = P.E
        S[sp + "_days"] = np.array([str(d) for d in P.day])
    S.update(vf=P.vf.astype(np.float32), cap=P.cap.astype(np.float32), lanes=P.lanes.astype(np.float32), length=P.length.astype(np.float32), on=P.on_attached, off=P.off_attached)
    np.savez_compressed(O + p + ".npz", **S); load.cache_clear(); print(p, f"{time.time() - t0:.0f}s", flush=True)
json.dump(dict(title="tfb-t1-img", id="telluriumrrr/tfb-t1-img", licenses=[dict(name="other")]), open(O + "dataset-metadata.json", "w"))
print("MB", sum(os.path.getsize(O + f) for f in os.listdir(O)) / 1e6)
