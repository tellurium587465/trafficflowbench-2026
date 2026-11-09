"""Domain token for sim-to-real (10/12): append a column 'is_synth' (1 = randomized twin window, 0 = real) to the
real+twin training data, and 0 to the real test data, so the model can keep separate 'fingerprint -> which bottleneck
activates' mappings for the two domains (the twin's differs, F57) while sharing what transfers (reading evidence and
state). nn_data_afdemsimR -> nn_data_afdemsimRd (81 columns), nn_test_afdem -> nn_test_afdemd (81 columns, all 0)."""
import os, shutil, numpy as np
for src, dst, synth in (("E:/tfb_nn/nn_data_afdemsimR", "E:/tfb_nn/nn_data_afdemsimRd", True),
                        ("E:/tfb_nn/nn_test_afdem", "E:/tfb_nn/nn_test_afdemd", False)):
    os.makedirs(dst, exist_ok=True)
    X = np.load(f"{src}/X.npy", mmap_mode="r"); off = np.load(f"{src}/off.npy")
    col = np.zeros(len(X), np.float32)
    if synth:
        date = np.load(f"{src}/date.npy", allow_pickle=True); syn = np.array([str(d)[:4] >= "2040" for d in date])
        for k in np.flatnonzero(syn):
            col[off[k]:off[k + 1]] = 1.0
        print(dst, "synthetic windows", int(syn.sum()), "rows share", round(float(col.mean()), 3), flush=True)
    out = np.lib.format.open_memmap(f"{dst}/X.npy", mode="w+", dtype=np.float32, shape=(int(len(X)), int(X.shape[1] + 1)))
    C = 1 << 21
    for a in range(0, len(X), C):
        b = min(a + C, len(X)); out[a:b, :-1] = X[a:b]; out[a:b, -1] = col[a:b]
    out.flush()
    assert np.array_equal(np.asarray(out[::997, :-1]), np.asarray(X[::997]), equal_nan=True), "existing columns changed"
    del out
    for f in os.listdir(src):
        if f == "names.npy":
            np.save(f"{dst}/names.npy", np.array([str(x) for x in np.load(f"{src}/names.npy", allow_pickle=True)] + ["is_synth"]))
        elif f != "X.npy":
            shutil.copyfile(f"{src}/{f}", f"{dst}/{f}")
    print("done", dst, flush=True)
