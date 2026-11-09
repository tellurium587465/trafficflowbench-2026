"""G261 (11/04, ledger #223, the strategy's 20:30 request): B12 - B10 and B11 - B10 on January / February in one flow (#179 10 needs the
January / February z of B12 itself). B's flow on the 316 selector ongoing windows as G197's "Bu" (B10 on January / February: X3a 0.2
after the networks, temperature 1.2, u2d 0.2, best_set; the masked-history context of G119), with the networks = capB0 fold 2 OOF
(probs_jfmh0__capB0, the only January / February family with four seeds, #107's proxy for 8 -> 16 seeds): b10j = seeds 0, 1 + best_set,
b11j = seeds 0-3 + best_set, b12j = seeds 0-3 + dex.DEX. Paired, cluster bootstrap (month x panel x day). Training split only.
usage: TFB_X3=x3af TFB_M3VER=v3 py -B -X utf8 g261_b12_janfeb.py"""
import os, sys, pickle, importlib.util
os.environ.setdefault("TFB_M3VER", "v3")
import numpy as np, pandas as pd
sys.path.insert(0, r"C:/Users/user/tfb/review"); import dex as _dx
R = r"C:/Users/user/tfb/"; a_ = sys.argv; sys.argv = ["g119", "run"]
spec = importlib.util.spec_from_file_location("g119", R + "review/g119_month_postproc.py"); h = importlib.util.module_from_spec(spec); spec.loader.exec_module(h); sys.argv = a_
T = h.T; W = h.W; X3 = 0.2; TE = 1.2
lg = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6))); tmp = lambda p: 1 / (1 + np.exp(-lg(p) / TE))
PC = [pickle.load(open(W + f"probs_jfmh0__capB0_s{s}_fold2030-12.pkl", "rb")) for s in range(4)]
X = [pickle.load(open("C:/tfb_data/kaggle_out/cap1/out/oof_x3a_s" + str(s) + ".pkl", "rb")) for s in (0, 1)]; U = [pickle.load(open(W + f"oof_u2d_s{s}.pkl", "rb")) for s in (0, 1)]
rows = []
for mo in h.MONTHS:
    ks = np.flatnonzero(T.META["is_sel"] & (h.MON == mo) & (T.META["cond"] == "queue_ongoing"))
    for p in sorted({str(T.META["panel"][k]) for k in ks}):
        P = h.load(p, "train"); key = (mo, p, "mask"); h.CTX.clear(); h.sim_blank.cache_clear(); h.CTX[key] = h.Ctx(key, P, np.where(P.blank, np.nan, P.V), P.V)
        for k in [k for k in ks if str(T.META["panel"][k]) == p and k in X[0] and k in X[1] and k in U[0] and str(k) in PC[0]]:
            t = int(T.META["t"][k]); lab, el = T.LAB[k]; y, e = lab.astype(bool), el.astype(bool); dom = int(P.times[t][8:10]) - 1
            px = (X[0][k] + X[1][k]) / 2; pu = (U[0][k] + U[1][k]) / 2; p2 = (PC[0][str(k)] + PC[1][str(k)]) / 2; p4 = sum(d[str(k)] for d in PC) / 4
            f2 = tmp(0.8 * ((1 - X3) * p2 + X3 * px) + 0.2 * pu); f4 = tmp(0.8 * ((1 - X3) * p4 + X3 * px) + 0.2 * pu)
            rows.append(dict(w=int(k), month=mo, panel=p, day=dom, b10j=h.iou(h.decode(f2), y, e, "queue_ongoing"), b11j=h.iou(h.decode(f4), y, e, "queue_ongoing"), b12j=h.iou(_dx.DEX(f4), y, e, "queue_ongoing")))
        h.load.cache_clear()
D = pd.DataFrame(rows); D.to_csv(W + "g261_jf.csv", index=False); cl = (D.month.astype(str) + "_" + D.panel + "_" + D.day.astype(str)).to_numpy()


def boot(d, cl, n=2000):
    ids = pd.factorize(cl)[0]; M_ = np.bincount(ids, d); N_ = np.bincount(ids); Rm = np.random.default_rng(0).multinomial(len(M_), np.full(len(M_), 1 / len(M_)), size=n); return float(((Rm @ M_) / (Rm @ N_)).std())


for b_ in ("b11j", "b12j"):
    d = (D[b_] - D.b10j).to_numpy(); se = boot(d, cl)
    print(f"{b_} - b10j, January / February {len(D)} windows: {d.mean():+.4f} (SE {se:.4f}, z {d.mean() / max(se, 1e-12):+.2f}) | Jan {d[(D.month == '2031-01').to_numpy()].mean():+.4f} Feb {d[(D.month == '2031-02').to_numpy()].mean():+.4f} | windows changed {int((np.abs(d) > 1e-12).sum())}", flush=True)
