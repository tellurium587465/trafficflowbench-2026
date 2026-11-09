"""G138 (10/31, ledger #98, record only): closes the procedure gap between G135 and G136. The fold-2 B0 seeds 0-3 on M3 v3 DEV ongoing with G135's procedure: production's days 1 .. D-1
of nn_data_march (t2_nn_finetune.finetune as t2_x1_official.py, no simulated blanks), no k. Day 1 keeps the base. Held-out days are not scored.
  fit <tag> : writes work/t2/x1_months/m3prod_<tag>.pkl = {window id: (base, x1)}
  score     : raw best_set and A's flow (seen, cone, far gate expected over k), x1 - base
usage: py g138_x1_march_prod.py fit capB0_s0_fold2030-12 | TFB_M3VER=v3 py g138_x1_march_prod.py score"""
import os, sys, time, pickle, importlib.util
os.environ.setdefault("TFB_M3VER", "v3")
MODE = (sys.argv[1:] or ["score"])[0]; R = r"C:/Users/user/tfb/"; W = R + "work/t2/"; OUT = W + "x1_months/"; sys.path.insert(0, R + "src")
TAGS = [f"capB0_s{s}_fold2030-12" for s in range(4)]
import numpy as np, pandas as pd


def module(nm, path):
    spec = importlib.util.spec_from_file_location(nm, path); mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


def boot(d, cl, n_boot=2000, seed=0):
    rng = np.random.default_rng(seed); ids = pd.factorize(cl)[0]; M = np.bincount(ids, d); N = np.bincount(ids)
    Rm = rng.multinomial(len(M), np.full(len(M), 1.0 / len(M)), size=n_boot); return float(((Rm @ M) / (Rm @ N)).std())


if MODE == "fit":
    import torch
    torch.set_num_threads(3); torch.set_num_threads = lambda n: None
    import t2_nn_finetune as FT
    tag = sys.argv[2]; t0 = time.time(); META = pickle.load(open(W + "m3v3_meta.pkl", "rb"))["meta"]; hold = lambda d: (d + 1) % 3 == 0
    byday = {}
    for m in META:
        if m["cond"] == "queue_ongoing" and not hold(m["day"]):
            byday.setdefault(m["day"], []).append(m["window_id"])
    base = pickle.load(open(W + f"probs_m3v3cap__{tag}.pkl", "rb")); res = {}
    for D0, wids in sorted(byday.items()):
        if D0 == 0:
            res.update({w: (base[w], base[w]) for w in wids}); continue
        net, mu, sd, cm, n = FT.finetune(tag, "nn_data_march", 1, D0, lr=2e-4, epochs=4, wpow=1.0); pr = FT.predict(net, mu, sd, cm, "nn_test_m3v3", wids); res.update({w: (base[w], pr[w]) for w in wids})
        open(OUT + f"g138_{tag}.log", "a").write(f"day {D0 + 1}: {n} windows, {time.time() - t0:.0f}s\n")
    pickle.dump(res, open(OUT + f"m3prod_{tag}.pkl", "wb")); open(OUT + f"g138_{tag}.log", "a").write(f"ALLDONE {time.time() - t0:.0f}s\n"); sys.exit()
g = module("g121", R + "review/g121_ongoing_oracle.py"); G, V, m3v2 = g.G, g.V, g.m3v2; best_set = g.best_set
raw = lambda p: best_set(p.ravel(), np.ones(p.size, bool)).reshape(p.shape)
PR = [pickle.load(open(OUT + f"m3prod_{t}.pkl", "rb")) for t in TAGS]; assert set(PR[0]) == set(g.DEV); print("seeds", len(PR), "windows", len(PR[0]), flush=True)
cols = {c: {} for c in ("raw_base", "raw_x1", "flow_base", "flow_x1")}
for w in g.DEV:
    m = G.META[w]; i = G.POS[w]; Xw = np.asarray(G.X[G.OFF[i]:G.OFF[i + 1]]); pl = g.FS.lgb_ongoing(Xw); kd = m3v2.kdist(m["day"]); cf = {k: g.cone_f(m, k) for k in kd}
    for nm, j in (("base", 0), ("x1", 1)):
        p = sum(d[w][j] for d in PR) / len(PR); cols["raw_" + nm][w] = G.iou(raw(p), m)
        cols["flow_" + nm][w] = sum(pr * G.iou(g.og_pred((0.8 * p + 0.2 * pl) * cf[k], m, k)[0], m) for k, pr in kd.items())
D = V.frame(cols).dropna(subset=["raw_x1"]); D.to_csv(OUT + "g138_x1_march_prod.csv", index=False); cl = D.panel + "_" + D.day.astype(str)
A = pd.read_csv(OUT + "g136_x1_march.csv").set_index("w"); print("identity base raw against G136 max |d|", float(np.max(np.abs(D.raw_base.to_numpy() - A.loc[D.w, "cap_raw_base"].to_numpy()))))
for c in ("raw", "flow"):
    x = (D[c + "_x1"] - D[c + "_base"]).to_numpy(); se = boot(x, cl); z = x.mean() / max(se, 1e-12); print(f"{c}: windows {len(D)} base {D[c + '_base'].mean():.4f} x1 - base {x.mean():+.4f} (SE {se:.4f}, z {z:+.2f}, /1.25 {z / 1.25:+.2f}) | windows changed {(np.abs(x) > 1e-9).sum()}")
x = D.raw_x1 - D.raw_base; print("by day (0-based):", " ".join(f"{lo}-{hi} {x[(D.day >= lo) & (D.day <= hi)].mean():+.4f}" for lo, hi in ((0, 6), (7, 13), (14, 20), (21, 31))))
