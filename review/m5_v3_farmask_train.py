"""M5 audit V3: the evaluator's far_mask inputs (nn_data_afdem_lastq.npz) vs nn_data X, window alignment with the
evaluator cache (featcache2_noT), label alignment, and the removal split of s34 by split x weekday."""
import sys, pickle, numpy as np, pandas as pd
sys.path.insert(0, "C:/Users/user/tfb/src")
W = "C:/Users/user/tfb/work/"
d = "E:/tfb_nn/nn_data_afdem/"
names = list(np.load(d + "names.npy", allow_pickle=True)); ci = names.index("last_queue")
off = np.load(d + "off.npy"); X = np.load(d + "X.npy", mmap_mode="r"); Y = np.load(d + "Y.npy", mmap_mode="r"); EL = np.load(d + "EL.npy", mmap_mode="r")
pn = np.load(d + "panel.npy", allow_pickle=True).astype(str); tt = np.load(d + "t.npy"); cond = np.load(d + "cond.npy", allow_pickle=True).astype(str)
date = np.load(d + "date.npy", allow_pickle=True).astype(str)
Z = np.load(W + "t2/featcache2_noT.npz", allow_pickle=True)
print("featcache files", Z.files)
for k, a in (("panel", pn), ("t", tt), ("cond", cond), ("date", date)):
    b = Z[k]; b = b.astype(str) if b.dtype.kind in "OU" else b
    print(f"  {k}: same length {len(b) == len(a)}, equal {bool(np.all(b == a)) if len(b) == len(a) else 'n/a'}")
print("  off equal", np.array_equal(Z["off"], off) if "off" in Z.files else "no off in featcache")
z = np.load("E:/tfb_nn/nn_data_afdem_lastq.npz"); lq, loff = z["lq"], z["off"]
print("lastq.npz: off equal", np.array_equal(loff, off), "lq rows", len(lq), "X rows", X.shape[0])
rng = np.random.default_rng(0); ks = rng.choice(len(off) - 1, 400, replace=False)
bad = 0
for k in ks:
    a, b = off[k], off[k + 1]
    if not np.array_equal(lq[a:b], np.asarray(X[a:b, ci]) == 1):
        bad += 1
print("sampled windows with lq != X[last_queue]==1:", bad, "/", len(ks))
LAB = pickle.load(open(W + "t2/labels_all2_noT.pkl", "rb"))
print("labels", len(LAB), "windows", len(off) - 1)
badl = bads = 0
for k in ks:
    a, b = off[k], off[k + 1]; lab, el = LAB[k]
    if lab.shape != (6, b - a):
        bads += 1; continue
    if not (np.array_equal(lab.astype(bool), np.asarray(Y[a:b]).T.astype(bool)) and np.array_equal(el.astype(bool), np.asarray(EL[a:b]).T.astype(bool))):
        badl += 1
print("label shape mismatches", bads, "label/Y or elig/EL mismatches", badl, "/", len(ks))
og = np.flatnonzero(cond == "queue_ongoing"); isel = Z["is_sel"] if "is_sel" in Z.files else None
nz = 0; farshare = []
for k in og:
    a, b = off[k], off[k + 1]; l = lq[a:b]
    if not l.any():
        nz += 1
    near = np.zeros(b - a, bool)
    for j in np.flatnonzero(l):
        near[max(0, j - 3):j + 4] = True
    farshare.append((~near).mean())
print(f"train ongoing windows {len(og)}, with no queue visible at T (all links far) {nz} ({nz/len(og):.2%}), mean far share {np.mean(farshare):.3f}")
if isel is not None:
    s = og[isel[og]]; nzs = sum(1 for k in s if not lq[off[k]:off[k + 1]].any())
    print(f"  selector-like ongoing windows {len(s)}, all-far {nzs}")
