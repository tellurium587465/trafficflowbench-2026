"""s_emp per segment (review B, 10/22; t2_model2.TFB_T2_SEMP=seg): write copies of NN data / test dirs with only the two
s_emp columns replaced, so the fix can be trained and scored without rebuilding everything.
  train : nn_data_afdem -> nn_data_afdem_semp   (windows from featcache2_noT_af: same order, off checked equal)
  m3    : nn_test_m3    -> nn_test_m3_semp      (windows from m3_meta.pkl, validation split)
  test  : nn_test_afdem -> nn_test_afdem_semp   (official windows from window_index.csv)
Identity first: the old definition recomputed here must equal the stored columns (every window, max |diff| printed).
X is copied on disk and patched through a memory map (low RAM).
usage: python g62_semp_patch.py <train|m3|test>"""
import os, sys, shutil, pickle, time
os.environ["TFB_T2_NOT"] = "1"; os.environ["TFB_T2_ALLFIN"] = "1"
sys.path.insert(0, r"C:/Users/user/tfb/src")
import numpy as np, pandas as pd
from cachelib import load, T2_PANELS
from common import NN, REL
from t2_model import PanelInputs
import t2_model2 as M2

kind = sys.argv[1]; t0 = time.time()
SRC = {"train": "nn_data_afdem", "m3": "nn_test_m3", "test": "nn_test_afdem"}[kind]; DST = SRC + "_semp"
names = [str(x) for x in np.load(f"{NN}/nn_data_afdem/names.npy", allow_pickle=True)]
IT, IH = names.index("s_emp_tail"), names.index("s_emp_head")
off = np.load(f"{NN}/{SRC}/off.npy")
if kind == "train":
    Z = np.load("C:/Users/user/tfb/work/t2/featcache2_noT_af.npz", allow_pickle=True)
    assert np.array_equal(Z["off"], off)
    wins = [(str(p), "train", int(t)) for p, t in zip(Z["panel"], Z["t"])]
elif kind == "m3":
    wid = [str(w) for w in np.load(f"{NN}/{SRC}/window_id.npy", allow_pickle=True)]
    meta = {m["window_id"]: m for m in pickle.load(open("C:/Users/user/tfb/work/t2/m3_meta.pkl", "rb"))["meta"]}
    wins = [(meta[w]["panel"], "validation", meta[w]["t"]) for w in wid]
else:
    wid = [str(w) for w in np.load(f"{NN}/{SRC}/window_id.npy", allow_pickle=True)]; loc = {}
    for sp in ("validation", "private"):
        for p in T2_PANELS:
            ti = {s: i for i, s in enumerate(load(p, sp).times)}
            for r in pd.read_csv(REL / "task2" / p / sp / "window_index.csv").itertuples():
                loc[r.window_id] = (p, sp, ti[r.forecast_origin])
    wins = [loc[w] for w in wid]
os.makedirs(f"{NN}/{DST}", exist_ok=True)
for f in os.listdir(f"{NN}/{SRC}"):
    shutil.copyfile(f"{NN}/{SRC}/{f}", f"{NN}/{DST}/{f}")
X = np.load(f"{NN}/{DST}/X.npy", mmap_mode="r+")
PIs = {}; err = 0.0; nchg = 0; big = 0
for i, (p, sp, t) in enumerate(wins):
    if (p, sp) not in PIs:
        PIs[(p, sp)] = PanelInputs(load(p, sp))
    PI = PIs[(p, sp)]; a, b = off[i], off[i + 1]
    M2.SEMP = ""; Xo, n = M2.kin_features(PI, t)
    old = np.column_stack([Xo[:, n.index("s_emp_tail")], Xo[:, n.index("s_emp_head")]]).astype(np.float32)
    ref = np.asarray(X[a:b][:, [IT, IH]])
    err = max(err, float(np.nanmax(np.where(np.isnan(ref) & np.isnan(old), 0, np.abs(ref - old)), initial=0.0)))
    M2.SEMP = "seg"; Xn, _ = M2.kin_features(PI, t)
    new = np.column_stack([Xn[:, n.index("s_emp_tail")], Xn[:, n.index("s_emp_head")]]).astype(np.float32)
    nchg += int((np.nan_to_num(new, nan=-999) != np.nan_to_num(old, nan=-999)).any()); big += int((np.abs(np.nan_to_num(old)) > 25).any())
    X[a:b, IT] = new[:, 0]; X[a:b, IH] = new[:, 1]
    if i % 5000 == 0:
        print(f"{i}/{len(wins)} {time.time() - t0:.0f}s", flush=True)
X.flush()
print(f"{kind}: {len(wins)} windows, identity of the old definition max |diff| {err}, windows changed {nchg}, "
      f"windows with |old| > 25 km/h {big} -> {DST} ({time.time() - t0:.0f}s)")
