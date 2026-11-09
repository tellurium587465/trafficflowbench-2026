"""G265 (11/04, ledger #225): January / February 316 selector ongoing windows, per window the spread of capB0's four seeds (mean over
cells of the variance across seeds 0-3, the proxy of the 40 nets' spread) and the IoU at the temperature 1.0 / 1.2 / 1.4 of
  A : G229's trow flow (capB0 s0-3, G119's SCG flow with the row T as the last queue), the temperature on the probabilities right before
      best_set (G119's decode patched, as G244)
  B : G261's B flow with seeds 0-3 (b11j: X3a 0.2, u2d 0.2, best_set), the temperature in place of 1.2
Training split only, writes g265_jf.csv. usage: TFB_M3VER=v3 py -B -X utf8 g265_jf_temps.py"""
import os, sys, pickle, importlib.util
os.environ.setdefault("TFB_M3VER", "v3")
import numpy as np, pandas as pd
R = r"C:/Users/user/tfb/"; a_ = sys.argv; sys.argv = ["g119", "run"]
spec = importlib.util.spec_from_file_location("g119", R + "review/g119_month_postproc.py"); g = importlib.util.module_from_spec(spec); spec.loader.exec_module(g); sys.argv = a_
T, MON, W = g.T, g.MON, g.W; X3 = 0.2
lg = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6))); sg = lambda x: 1 / (1 + np.exp(-x)); te_ = lambda p, te: p if te == 1.0 else sg(lg(p) / te)
PC = [pickle.load(open(W + f"probs_jfmh0__capB0_s{s}_fold2030-12.pkl", "rb")) for s in range(4)]
X = [pickle.load(open("C:/tfb_data/kaggle_out/cap1/out/oof_x3a_s" + str(s) + ".pkl", "rb")) for s in (0, 1)]; U = [pickle.load(open(W + f"oof_u2d_s{s}.pkl", "rb")) for s in (0, 1)]
dec0 = g.decode; KN = [1.0]; g.decode = lambda pb: dec0(te_(pb, KN[0])); TES = (1.0, 1.2, 1.4); rows = []
KS = [k for k in np.flatnonzero(T.META["is_sel"] & np.isin(MON, g.MONTHS)) if str(T.META["cond"][k]) == "queue_ongoing"]
for mo in g.MONTHS:
    for p in sorted({str(T.META["panel"][k]) for k in KS if MON[k] == mo}):
        P = g.load(p, "train"); key = (mo, p, "mask"); g.CTX.clear(); g.sim_blank.cache_clear(); g.CTX[key] = g.Ctx(key, P, np.where(P.blank, np.nan, P.V), P.V); c = g.CTX[key]
        for k in [k for k in KS if MON[k] == mo and str(T.META["panel"][k]) == p]:
            t = int(T.META["t"][k]); dom = int(P.times[t][8:10]) - 1; lab, el = T.LAB[k]; y, e = lab.astype(bool), el.astype(bool)
            S4 = np.stack([d[str(k)] for d in PC]); p_ = S4.mean(axis=0); r = dict(w=int(k), month=mo, panel=p, day=dom, spread=float(S4.var(axis=0).mean()))
            lq = np.asarray(g.XN[g.OFFN[k]:g.OFFN[k + 1]][:, g.G.CI], float) == 1; ot = c.obs[t].astype(bool); lq2 = np.where(ot, c.q[t].astype(bool), lq); cone = g.cone_parts(c, t, lq2, P.length)
            for te in TES:
                KN[0] = te; r[f"A{te}"] = sum(pr * g.iou(g.ongoing_pred(c, p_ * g.cone_f(c, cone, t, kk), t, kk, True, lq2), y, e, "queue_ongoing") for kk, pr in g.m3v2.kdist(dom, cap=3).items())
            KN[0] = 1.0
            if k in X[0] and k in X[1] and k in U[0]:
                f_ = 0.8 * ((1 - X3) * p_ + X3 * (X[0][k] + X[1][k]) / 2) + 0.2 * (U[0][k] + U[1][k]) / 2
                for te in TES:
                    r[f"B{te}"] = g.iou(dec0(te_(f_, te)), y, e, "queue_ongoing")
            rows.append(r)
        g.load.cache_clear()
g.decode = dec0; D = pd.DataFrame(rows); D.to_csv(W + "g265_jf.csv", index=False); print(len(D), "windows written g265_jf.csv; with B", int(D["B1.2"].notna().sum()), flush=True)
