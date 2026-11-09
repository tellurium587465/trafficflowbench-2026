"""In-distribution check inside the twin: month-CV OOF of world A's selector windows (is_sel) for a model trained on
world A only and one trained on A + randomized worlds (the R windows are always on the training side). Windows are
matched across the two data sets by (panel, t, cond, date). Plain mean IoU (expected-IoU set, onset at T+30 only)."""
import sys, pickle, numpy as np
def load(d, tag):
    D = {f: np.load(f"E:/tfb_nn/{d}/{f}.npy", allow_pickle=True, mmap_mode="r") for f in ("off", "panel", "t", "cond", "date", "is_sel")}
    Y = np.load(f"E:/tfb_nn/{d}/Y.npy", mmap_mode="r"); E = np.load(f"E:/tfb_nn/{d}/EL.npy", mmap_mode="r")
    oof = pickle.load(open(f"C:/Users/user/tfb/work/t2/oof_{tag}.pkl", "rb")); out = {}
    for k, p in oof.items():
        if not D["is_sel"][k]:
            continue
        a, b = D["off"][k], D["off"][k + 1]; y = np.asarray(Y[a:b]).T.astype(bool); el = np.asarray(E[a:b]).T.astype(bool)
        p = p.copy(); c = str(D["cond"][k])
        if c == "queue_onset":
            p[:-1] = 0
        pe = p.ravel(); o = np.argsort(-pe); cs = np.cumsum(pe[o]); kk = np.arange(1, pe.size + 1)
        kb = int(np.argmax(cs / (kk + pe.sum() - cs))) + 1; s = np.zeros(pe.size, bool); s[o[:kb]] = True; s = s.reshape(p.shape)
        u = ((s | y) & el).sum(); out[(str(D["panel"][k]), int(D["t"][k]), c, str(D["date"][k]))] = 1.0 if u == 0 else (s & y & el).sum() / u
    return out
A = load("nn_data_twinA", sys.argv[1]); B = load("nn_data_twinAR", sys.argv[2]); ks = [k for k in A if k in B]
for c in ("queue_onset", "queue_ongoing"):
    d = np.array([B[k] - A[k] for k in ks if k[2] == c]); a = np.mean([A[k] for k in ks if k[2] == c])
    print(f"{c}: A-only {a:.4f}, A+R - A-only {d.mean():+.4f} (SE {d.std() / np.sqrt(len(d)):.4f}, n {len(d)})")
