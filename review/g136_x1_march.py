"""G136 (10/31, ledger #96): the networks of G135 on March. X1 of the fold-2 B0 seeds 0-3 on M3 v3 DEV ongoing by the M3 path (src/t2_m3v2_x1.py unchanged, 3 threads), so one
family of networks is read on January, February and March. The production pair c24f_attnglob s0 / s1 (saved X1 files) is scored beside it as the reference.
  dry       : 5 windows: base prediction against the stored probabilities, score identity against G108 column A, one fine-tune timed, NaN share of the fine-tune pools
  fit <tag> : writes work/t2/probs_m3v3v2x1__<tag>.pkl; progress in work/t2/x1_months/g136_<tag>.log
  score     : raw best_set and A's flow, expected over k, x1 - base per family, by day band, and split at base IoU 0.95 (record only)
usage: TFB_M3VER=v3 py g136_x1_march.py dry | fit capB0_s0_fold2030-12 | score"""
import os, sys, time, pickle, shutil, runpy, importlib.util
os.environ.setdefault("TFB_M3VER", "v3")
MODE = (sys.argv[1:] or ["score"])[0]; R = r"C:/Users/user/tfb/"; W = R + "work/t2/"; OUT = W + "x1_months/"; sys.path.insert(0, R + "src")
TAGS = {"cap": [f"capB0_s{s}_fold2030-12" for s in range(4)], "prod": [f"c24f_attnglob_s{s}" for s in (0, 1)]}
if MODE == "fit":
    import torch
    torch.set_num_threads(3); torch.set_num_threads = lambda n: None
    import t2_nn_finetune as FT
    tag = sys.argv[2]; f = W + f"probs_m3v3__{tag}.pkl"; t0 = time.time(); n = [0]; ft = FT.finetune
    if not os.path.exists(f):                                   # the name the M3 path reads; no production file carries this tag
        shutil.copyfile(W + f"probs_m3v3cap__{tag}.pkl", f)

    def counted(*a, **k):
        r = ft(*a, **k); n[0] += 1
        if n[0] % 15 == 0:
            open(OUT + f"g136_{tag}.log", "a").write(f"{n[0]} of 165, {time.time() - t0:.0f}s\n")
        return r


    FT.finetune = counted; sys.argv = ["t2_m3v2_x1.py", tag]; runpy.run_path(R + "src/t2_m3v2_x1.py", run_name="__main__")
    open(OUT + f"g136_{tag}.log", "a").write(f"ALLDONE {time.time() - t0:.0f}s\n"); sys.exit()
import numpy as np, pandas as pd


def module(nm, path):
    spec = importlib.util.spec_from_file_location(nm, path); mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


def boot(d, cl, n_boot=2000, seed=0):
    rng = np.random.default_rng(seed); ids = pd.factorize(cl)[0]; M = np.bincount(ids, d); N = np.bincount(ids)
    Rm = rng.multinomial(len(M), np.full(len(M), 1.0 / len(M)), size=n_boot); return float(((Rm @ M) / (Rm @ N)).std())


g = module("g121", R + "review/g121_ongoing_oracle.py"); G, V, m3v2 = g.G, g.V, g.m3v2; best_set = g.best_set
raw = lambda p: best_set(p.ravel(), np.ones(p.size, bool)).reshape(p.shape)


def score(wins, XS):
    cols = {}
    for f in XS:
        for c in ("raw_base", "raw_x1", "flow_base", "flow_x1"):
            cols[f"{f}_{c}"] = {}
    for w in wins:
        m = G.META[w]; i = G.POS[w]; Xw = np.asarray(G.X[G.OFF[i]:G.OFF[i + 1]]); pl = g.FS.lgb_ongoing(Xw); kd = m3v2.kdist(m["day"]); cf = {k: g.cone_f(m, k) for k in kd}
        for f, X in XS.items():
            b = X[(w, 0)]; cols[f"{f}_raw_base"][w] = G.iou(raw(b), m)
            cols[f"{f}_raw_x1"][w] = sum(pr * G.iou(raw(X[(w, k)]), m) for k, pr in kd.items())
            cols[f"{f}_flow_base"][w] = sum(pr * G.iou(g.og_pred((0.8 * b + 0.2 * pl) * cf[k], m, k)[0], m) for k, pr in kd.items())
            cols[f"{f}_flow_x1"][w] = sum(pr * G.iou(g.og_pred((0.8 * X[(w, k)] + 0.2 * pl) * cf[k], m, k)[0], m) for k, pr in kd.items())
    return cols


def family(f):
    ds = [pickle.load(open(W + f"probs_m3v3v2x1__{t}.pkl", "rb")) for t in TAGS[f] if os.path.exists(W + f"probs_m3v3v2x1__{t}.pkl")]
    return {key: sum(d[key] for d in ds) / len(ds) for key in ds[0]}, len(ds)


ref = pd.read_csv(W + "g108_rectangles.csv").set_index("w")["A"]
if MODE == "dry":
    import torch
    torch.set_num_threads(3); torch.set_num_threads = lambda n: None
    import t2_nn_finetune as FT
    from common import NN
    w0 = g.DEV[:5]; tag = TAGS["cap"][0]; net, mu, sd, cm = FT.CI.load_model(tag); P = pickle.load(open(W + f"probs_m3v3cap__{tag}.pkl", "rb")); pr = FT.predict(net, mu, sd, cm, "nn_test_m3v3", w0)
    print("base prediction against probs_m3v3cap max |dp|", max(float(np.abs(pr[w] - P[w]).max()) for w in w0))
    X, n = family("prod"); c = score(w0, {"prod": X}); print("score identity flow_x1 against G108 A, 5 windows, max |d|", max(abs(c["prod_flow_x1"][w] - ref[w]) for w in w0))
    t0 = time.time(); r = FT.finetune(tag, "nn_data_march", 8, 10, lr=2e-4, epochs=4); print("one fine-tune k = 3:", r[4], "windows", f"{time.time() - t0:.1f}s", flush=True)
    AF = FT.data("nn_data_afdem"); MA = FT.data("nn_data_march"); isel = np.load(f"{NN}/nn_data_afdem/is_sel.npy"); mon = np.array([str(d)[:7] for d in AF["date"]]); rng = np.random.default_rng(0)


    def nan_share(D, ks):
        ks = rng.choice(ks, min(300, len(ks)), replace=False); v = [np.array(D["X"][int(D["off"][k]):int(D["off"][k + 1])], dtype=np.float32) for k in ks]; v = np.concatenate(v); return float((~np.isfinite(v)).mean())


    for mo in ("2031-01", "2031-02"):
        print(mo, "NaN share: all windows", round(nan_share(AF, np.flatnonzero(mon == mo)), 4), "| selector-like", round(nan_share(AF, np.flatnonzero((mon == mo) & isel)), 4))
    print("nn_data_march all windows", round(nan_share(MA, np.arange(len(MA["off"]) - 1)), 4))
    TX = np.load(f"{NN}/nn_test_m3v3/X.npy", mmap_mode="r"); TO = np.load(f"{NN}/nn_test_m3v3/off.npy"); TW = [str(x) for x in np.load(f"{NN}/nn_test_m3v3/window_id.npy", allow_pickle=True)]; pos = {x: i for i, x in enumerate(TW)}
    print("nn_test_m3v3 DEV ongoing", round(float((~np.isfinite(np.concatenate([np.array(TX[TO[pos[w]]:TO[pos[w] + 1]], dtype=np.float32) for w in g.DEV]))).mean()), 4)); sys.exit()
XS = {}
for f in TAGS:
    XS[f], n = family(f); print(f, "seeds", n, flush=True)
cols = score(g.DEV, XS); D = V.frame(cols).dropna(subset=["prod_raw_x1"]); D.to_csv(OUT + "g136_x1_march.csv", index=False); cl = D.panel + "_" + D.day.astype(str)
print(len(D), "windows | identity prod flow_x1 against G108 A max |d|", f"{float(np.max(np.abs(D.prod_flow_x1 - ref.loc[D.w].to_numpy()))):.2e}", flush=True)
for f in TAGS:
    for c in ("raw", "flow"):
        x = (D[f"{f}_{c}_x1"] - D[f"{f}_{c}_base"]).to_numpy(); se = boot(x, cl); z = x.mean() / max(se, 1e-12)
        print(f"{f} {c}: base {D[f'{f}_{c}_base'].mean():.4f} x1 - base {x.mean():+.4f} (SE {se:.4f}, z {z:+.2f}, /1.25 {z / 1.25:+.2f}) | windows changed {(np.abs(x) > 1e-9).sum()}")
    x = D[f"{f}_raw_x1"] - D[f"{f}_raw_base"]
    print(" ", f, "by day (0-based):", " ".join(f"{lo}-{hi} {x[(D.day >= lo) & (D.day <= hi)].mean():+.4f}" for lo, hi in ((0, 6), (7, 13), (14, 20), (21, 31))), "| base IoU >= 0.95:", f"{x[D[f'{f}_raw_base'] >= 0.95].mean():+.4f} ({(D[f'{f}_raw_base'] >= 0.95).sum()})", "< 0.95:", f"{x[D[f'{f}_raw_base'] < 0.95].mean():+.4f}")
C = pd.read_csv(OUT + "g135_x1_months.csv"); C["x"] = C.raw_x1 - C.raw_base
print("record: January / February split at base IoU 0.95:", " ".join(f"{mo} >= 0.95 {d.x[d.raw_base >= 0.95].mean():+.4f} ({(d.raw_base >= 0.95).sum()}) < 0.95 {d.x[d.raw_base < 0.95].mean():+.4f}" for mo, d in C.groupby("month")))
