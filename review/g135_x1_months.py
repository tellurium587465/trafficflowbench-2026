"""G135 (10/31, ledger #95, strategy, PUBLIC_PRIVATE item 4): does X1 keep its sign on other months? Rolling origin on 2031-01 / 2031-02 with networks that never saw
them (fold-2 B0, seeds 0-3): for each day D, X1 as in production (t2_nn_finetune.finetune: lr 2e-4, 4 epochs, seed 0, days 1 .. D-1 of the same month) on the
month's windows of the earlier days, with the same number of windows March would give (as the V9 placebo), then predict day D's selector-like ongoing windows;
day 1 keeps the base. Base = the same network on the CPU, no fine-tuning.
  fit <tag> : writes work/t2/x1_months/<tag>.pkl = {window index: (base, x1)}
  score     : raw best_set and the SCG flow (G119 path), x1 - base per month, 4 seed mean
usage: py g135_x1_months.py fit capB0_s0_fold2030-12 | TFB_M3VER=v3 py g135_x1_months.py score"""
import os, sys, time, pickle, importlib.util
MODE = (sys.argv[1:] or ["score"])[0]; R0 = r"C:/Users/user/tfb/"; OUT = R0 + "work/t2/x1_months/"; os.makedirs(OUT, exist_ok=True); MONTHS = ("2031-01", "2031-02")
import numpy as np
sys.path.insert(0, R0 + "src")
if MODE == "fit":
    import torch
    torch.set_num_threads(3); torch.set_num_threads = lambda n: None
    import t2_nn_finetune as FT
    from common import NN
    tag = sys.argv[2]; f = OUT + tag + ".pkl"; t0 = time.time()
    D = FT.data("nn_data_afdem"); MD = FT.data("nn_data_march"); isel = np.load(f"{NN}/nn_data_afdem/is_sel.npy"); cond = np.load(f"{NN}/nn_data_afdem/cond.npy", allow_pickle=True); mon = np.array([str(d)[:7] for d in D["date"]])
    nm = lambda d: int(((MD["day"] >= 1) & (MD["day"] <= d - 1)).sum())


    def pred(net, mu, sd, cm, ks):
        out = {}
        with torch.no_grad():
            for k in ks:
                a, b = int(D["off"][k]), int(D["off"][k + 1]); v = torch.from_numpy(np.array(D["X"][a:b], dtype=np.float32))[None]
                out[int(k)] = torch.sigmoid(net(FT.enc(v, cm, mu, sd), torch.ones((1, b - a)))[..., :6]).numpy()[0].T.copy()
        return out


    net0, mu0, sd0, cm0 = FT.CI.load_model(tag); res = {}; nft = 0
    for mo in MONTHS:
        ks = np.flatnonzero(isel & (mon == mo) & (cond == "queue_ongoing")); base = pred(net0, mu0, sd0, cm0, ks)
        for d in sorted(set(D["day"][ks])):
            kd = ks[D["day"][ks] == d]
            if d == 1:
                x1 = {int(k): base[int(k)] for k in kd}
            else:
                net, mu, sd, cm, n = FT.finetune(tag, "nn_data_afdem", 1, d - 1, lr=2e-4, epochs=4, wpow=1.0, month=mo, max_n=nm(d)); x1 = pred(net, mu, sd, cm, kd); nft += 1
            res.update({k: (base[k], x1[k]) for k in x1})
        print(tag, mo, "done", nft, f"{time.time() - t0:.0f}s", flush=True)
    pickle.dump(res, open(f, "wb")); print("ALLDONE", tag, f"{time.time() - t0:.0f}s", flush=True); sys.exit()
os.environ.setdefault("TFB_M3VER", "v3")
import pandas as pd
from collections import defaultdict
spec = importlib.util.spec_from_file_location("g119", R0 + "review/g119_month_postproc.py"); g = importlib.util.module_from_spec(spec); spec.loader.exec_module(g); T, G, m3v2 = g.T, g.G, g.m3v2
TAGS = [f"capB0_s{s}_fold2030-12" for s in range(4) if os.path.exists(OUT + f"capB0_s{s}_fold2030-12.pkl")]; PR = [pickle.load(open(OUT + t + ".pkl", "rb")) for t in TAGS]; print("seeds", len(PR), flush=True)
OOF = [pickle.load(open(rf"C:/tfb_data/kaggle_out/cap1/out/oof_capB0_s{s}.pkl", "rb")) for s in range(len(PR))]; print("base CPU vs stored GPU OOF max |dp|", round(max(float(np.abs(PR[i][k][0] - OOF[i][k]).max()) for i in range(len(PR)) for k in list(PR[0])[:20]), 4)); rows = []
for mo in MONTHS:
    ks = sorted(k for k in PR[0] if g.MON[k] == mo)
    for p in sorted({str(T.META["panel"][k]) for k in ks}):
        P = g.load(p, "train"); key = (mo, p, "mask"); g.CTX.clear(); g.sim_blank.cache_clear()
        g.CTX[key] = g.Ctx(key, P, np.where(P.blank, np.nan, P.V), P.V); c = g.CTX[key]
        for k in [k for k in ks if str(T.META["panel"][k]) == p]:
            t = int(T.META["t"][k]); lab, el = T.LAB[k]; y, e = lab.astype(bool), el.astype(bool); lq = np.asarray(g.XN[g.OFFN[k]:g.OFFN[k + 1]][:, G.CI], float) == 1; dom = int(P.times[t][8:10]) - 1
            cone = g.cone_parts(c, t, lq, P.length); r = defaultdict(float, w=int(k), month=mo, panel=p, day=dom)
            for nm_, i in (("base", 0), ("x1", 1)):
                pm = sum(d[k][i] for d in PR) / len(PR); r["raw_" + nm_] = g.iou(g.decode(pm), y, e, "queue_ongoing")
                for kk, pr in m3v2.kdist(dom, cap=3).items():
                    r["scg_" + nm_] += pr * g.iou(g.ongoing_pred(c, pm * g.cone_f(c, cone, t, kk), t, kk, True, lq), y, e, "queue_ongoing")
            rows.append(dict(r))
        g.load.cache_clear()
D = pd.DataFrame(rows); D.to_csv(OUT + "g135_x1_months.csv", index=False)


def boot(d, cl, n_boot=2000, seed=0):
    rng = np.random.default_rng(seed); ids = pd.factorize(cl)[0]; M = np.bincount(ids, d); N = np.bincount(ids)
    Rm = rng.multinomial(len(M), np.full(len(M), 1.0 / len(M)), size=n_boot); return float(((Rm @ M) / (Rm @ N)).std())


for mo in MONTHS:
    d = D[D.month == mo]; cl = d.panel + "_" + d.day.astype(str)
    for nm_ in ("raw", "scg"):
        x = (d[nm_ + "_x1"] - d[nm_ + "_base"]).to_numpy(); se = boot(x, cl); print(f"{mo} {nm_}: windows {len(d)} base {d[nm_ + '_base'].mean():.4f} x1 - base {x.mean():+.4f} (SE {se:.4f}, z {x.mean() / max(se, 1e-12):+.2f}, ÷1.25 {x.mean() / max(se, 1e-12) / 1.25:+.2f}) | windows changed {(np.abs(x) > 1e-9).sum()}")
