"""G236 (11/03, ledger #189 -> A13): the April dense pool's identity. nn_data_april_sp1 = build_march.py 1 private; the build's greedy
rule (per panel and condition, keep an origin >= s rows after the last kept) at s = 6 must give nn_data_april bit for bit and at s = 3
nn_data_april_sp3 (build_march.py 3 private) bit for bit, as G222 for March. usage: py -B -X utf8 g236_april_pool_id.py"""
import numpy as np
NN = "C:/tfb_data/tfb_nn"; D1 = "nn_data_april_sp1"
M = {f: np.load(f"{NN}/{D1}/{f}.npy", allow_pickle=True) for f in ("panel", "t", "cond", "date", "off")}


def keep(sp):
    out, last = [], {}
    for i, (p, t, c) in enumerate(zip(M["panel"], M["t"].astype(int), M["cond"])):
        if t - last.get((p, c), -10 ** 9) >= sp:
            out.append(i); last[(p, c)] = t
    return np.array(out, int)


for sp, ref in ((6, "nn_data_april"), (3, "nn_data_april_sp3")):
    k = keep(sp); R = {f: np.load(f"{NN}/{ref}/{f}.npy", allow_pickle=True) for f in ("panel", "t", "cond", "date")}
    meta = len(k) == len(R["t"]) and all(np.array_equal(np.asarray(M[f][k]).astype(str), np.asarray(R[f]).astype(str)) for f in ("panel", "t", "cond", "date"))
    ok = meta
    if meta:
        rows = np.concatenate([np.arange(M["off"][i], M["off"][i + 1]) for i in k])
        for f in ("X", "Y", "EL"):
            a = np.load(f"{NN}/{D1}/{f}.npy", mmap_mode="r"); b = np.load(f"{NN}/{ref}/{f}.npy", mmap_mode="r")
            ok &= bool(np.array_equal(np.asarray(a[rows]), np.asarray(b), equal_nan=True))
    print(f"spacing {sp}: {len(k)} windows from {D1} vs {ref} {len(R['t'])}: meta {meta}, X/Y/EL equal {ok}", flush=True)
