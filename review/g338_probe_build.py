"""G338 (11/05, ledger #270): the two one-sided blank probes on finA21's March (validation) rows. (a) ongoing windows whose box is young or
small (G313 / G337's ys, known at T from R of nn_test_afdem), (b) onset windows on day 1-2 of the month (forecast_origin). The queue_pred of
those windows' rows -> 0; every other byte of finA21's submission.csv unchanged (checked). April rows untouched.
Writes work/subs/probe270a.zip, probe270b.zip and work/t2/g338_windows.csv. usage: py -B -X utf8 g338_probe_build.py"""
import sys, zipfile, io, hashlib
import numpy as np, pandas as pd
R0 = "C:/Users/user/tfb/"; sys.path.insert(0, r"C:/Users/user/tfb_a8/src")
from cachelib import T2_PANELS
from common import REL
NN = "C:/tfb_data/tfb_nn/nn_test_afdem/"; R = np.load(NN + "R.npy", mmap_mode="r"); off = np.load(NN + "off.npy"); pos = {str(w): i for i, w in enumerate(np.load(NN + "window_id.npy", allow_pickle=True))}


def ys_of(Rw):   # identical to g337's r_attr (young or small last box)
    o = Rw[:, :12, 2].astype(float) > 0; v = Rw[:, :12, 0].astype(float); q = o & (v <= 1.0); rows = np.flatnonzero(o.any(axis=0))
    if not len(rows):
        return True
    ql = q[:, rows[-1]]; n = int(ql.sum()); age = 0
    for r in rows[::-1]:
        if (q[:, r] & ql).any():
            age += 1
        else:
            break
    return n <= 3 or age <= 3


rows = []
for p in T2_PANELS:
    for r in pd.read_csv(REL / "task2" / p / "validation" / "window_index.csv").itertuples():
        i = pos[r.window_id]; rows.append(dict(window_id=r.window_id, cond=r.condition, dom=pd.Timestamp(r.forecast_origin).day, ys=ys_of(np.asarray(R[off[i]:off[i + 1]]))))
Wd = pd.DataFrame(rows); Wd["a"] = (Wd.cond == "queue_ongoing") & Wd.ys; Wd["b"] = (Wd.cond == "queue_onset") & (Wd.dom <= 2); Wd.to_csv(R0 + "work/t2/g338_windows.csv", index=False)
print(f"(a) ongoing young/small: {int(Wd.a.sum())} of {int((Wd.cond == 'queue_ongoing').sum())} | (b) onset day 1-2: {int(Wd.b.sum())} of {int((Wd.cond == 'queue_onset').sum())}", flush=True)
K = pd.read_csv(REL / "submission_key.csv", dtype=str, usecols=["submission_id", "task", "split", "window_id"]); K = K[(K.task == "queue") & (K.split == "validation")]
src = zipfile.ZipFile(R0 + "work/subs/finA21.zip").read("submission.csv"); print("finA21 csv sha", hashlib.sha256(src).hexdigest()[:8], flush=True)
lines = src.split(b"\n")
for tag in ("a", "b"):
    ids = set(K[K.window_id.isin(Wd.window_id[Wd[tag]])].submission_id.astype(str).str.encode("ascii")); out = []; n1 = nrow = 0
    for ln in lines:
        c = ln.split(b",", 1)[0]
        if c in ids:
            f = ln.split(b","); assert f[1] == b"queue"; nrow += 1; n1 += f[4] == b"1"; f[4] = b"0"; ln = b",".join(f)
        out.append(ln)
    body = b"\n".join(out); assert len(out) == len(lines) and nrow == len(ids)
    diff = sum(a_ != b_ for a_, b_ in zip(lines, out)); zp = R0 + f"work/subs/probe270{tag}.zip"
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("submission.csv", body)
    print(f"probe ({tag}): rows of the windows {nrow}, predicted-1 cells set to 0: {n1}, lines changed {diff}, csv sha {hashlib.sha256(body).hexdigest()[:8]} -> {zp}", flush=True)
