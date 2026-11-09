"""G146 (10/31, ledger #107): B's April ongoing over 16 seeds instead of 8, a variance-only change (B7 + TFB_T2_OG_SEEDS_PRIV=16).
  m3    : M3 v3 DEV ongoing (held-out days are not read), raw best_set on the CF5 weighted blend of q24f seeds 0-7 (B8) and 0-15 (B16),
          from work/t2/probs_m3v3__q24f_<c>_s<s>.pkl (t2_nn_cpu_infer.py). Writes work/t2/g146_seed16.csv for judge.py --kind model --k 13.
  proxy : the January / February stand-in of the ledger row, fold 2 B0 (Kaggle cap1) seeds {0,1} against {0,1,2,3}, raw IoU, out-of-fold.
  dry   : 5 dev onset windows, the 8-seed blend written here against the cached final blend of G120.
usage: TFB_M3VER=v3 py g146_seed16.py [m3 | proxy | dry] [out dir of cap1]"""
import os, sys, pickle
MODE = (sys.argv[1:] or ["m3"])[0]; KO = (sys.argv[2:] or [r"C:/tfb_data/kaggle_out/cap1/out"])[0]
os.environ.setdefault("TFB_M3VER", "v3")
import numpy as np, pandas as pd
sys.path.insert(0, r"C:/Users/user/tfb/review")
import g130_cap_probe as C
R = C.R; CF5 = (("attnglob", 2), ("attnwp05", 2), ("attn1", 2), ("drop1", 1), ("r13", 1))


def blend(g, seeds, ws):
    W = g.W; PR = [(pickle.load(open(W + f"probs_{g.G.PRE}__q24f_{c}_s{s}.pkl", "rb")), wt) for s in seeds for c, wt in CF5]; sw = sum(wt for _, wt in PR)
    return {w: sum(wt * p[w] for p, wt in PR) / sw for w in ws}


def dry():
    g = C.module("g120", R + "review/g120_onset_gate.py"); PN = g.nn_blend(); n0 = g.DEV[:5]; P = blend(g, range(8), n0)
    print("identity 8-seed blend against G120 (5 windows): max |dp|", max(float(np.abs(P[w] - PN[w]).max()) for w in n0), flush=True)


def m3():
    g = C.module("g121", R + "review/g121_ongoing_oracle.py"); G, V = g.G, g.V; best_set = C.module("g120", R + "review/g120_onset_gate.py").best_set; raw = lambda p: best_set(p.ravel(), np.ones(p.size, bool)).reshape(p.shape)
    P8 = blend(g, range(8), g.DEV); P16 = blend(g, range(16), g.DEV); cols = {"B8": {}, "B16": {}}
    for w in g.DEV:
        m = G.META[w]; cols["B8"][w] = G.iou(raw(P8[w]), m); cols["B16"][w] = G.iou(raw(P16[w]), m)
    D = V.frame(cols).dropna(subset=["B8"]); D.to_csv(R + "work/t2/g146_seed16.csv", index=False); x = (D.B16 - D.B8).to_numpy(); print(len(D), "windows B8", round(D.B8.mean(), 4), "B16", round(D.B16.mean(), 4), f"diff {x.mean():+.4f}", flush=True)


def proxy():
    g = C.module("g119", R + "review/g119_month_postproc.py"); T = g.T; S = {"s2": (0, 1), "s4": (0, 1, 2, 3)}; P = {n: C.seedmean([f"{KO}/oof_capB0_s{s}.pkl" for s in v]) for n, v in S.items()}
    print("seeds found", {n: P[n][1] for n in P}, flush=True); P = {n: P[n][0] for n in P}; rows = []
    for k in np.flatnonzero(T.META["is_sel"] & np.isin(g.MON, g.MONTHS)):
        if k in P["s2"] and str(T.META["cond"][k]) == "queue_ongoing":
            lab, el = T.LAB[k]; r = dict(w=int(k), month=g.MON[k], panel=str(T.META["panel"][k]), day=str(T.META["date"][k])[:10])
            for n in P:
                r[n] = g.iou(g.decode(P[n][k]), lab.astype(bool), el.astype(bool), "queue_ongoing")
            rows.append(r)
    D = pd.DataFrame(rows); D.to_csv(R + "work/t2/g146_proxy.csv", index=False)
    for mo in g.MONTHS:
        d = D[D.month == mo]; x = (d.s4 - d.s2).to_numpy(); se = C.boot(x, d.panel + "_" + d.day); print(mo, len(d), "2 seeds", round(d.s2.mean(), 4), "4 seeds", round(d.s4.mean(), 4), f"diff {x.mean():+.4f} (SE {se:.4f}, z {x.mean() / max(se, 1e-12):+.2f})", flush=True)


if __name__ == "__main__":
    {"m3": m3, "proxy": proxy, "dry": dry}[MODE]()
