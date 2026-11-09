"""G175 (11/01, ledger #124): the bundle of #121 in the 11/4 read. A side: final A7's March ongoing flow against the same + same-day X1 + X3a 0.2 + temperature 1.2. B side: B7's April ongoing flow (q24f CF5 8 seeds -> seen x0.85 -> best_set -> far gate, no X1, cone, LightGBM) against
0.8 q24f + 0.2 X3a -> seen -> temperature 1.2 -> best_set -> far gate.
  dev  : M3 dev ongoing, identity A and bundle against g171, the B side, January and February for the B side; readable now
  read : hold-out ongoing only, one read after #63. Needs TFB_READ_1104=1. Merges each side with its dev value by inverse variance; line = merged z / 1.25 >= +2 (k = 1) and January, February >= 0
usage: TFB_M3VER=v3 py -B -X utf8 g175_bundle_read.py dev | read"""
import os, sys, glob, pickle, importlib.util
os.environ.setdefault("TFB_M3VER", "v3"); sys.dont_write_bytecode = True; RUN = sys.argv[1]; R0 = r"C:/Users/user/tfb/"
if RUN == "read":
    assert os.environ.get("TFB_READ_1104") == "1", "the hold-out read is 11/4, after #63, once"
import numpy as np, pandas as pd


def module(nm, path):
    spec = importlib.util.spec_from_file_location(nm, path); mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


sys.argv = [sys.argv[0], "run"]; g = module("g121", R0 + "review/g121_ongoing_oracle.py"); g132 = module("g132", R0 + "review/g132_p3_grid.py")
G, V, m3v2, best_set = g.G, g.V, g.m3v2, g.best_set; W = g.W; TE = 1.2; X3 = 0.2
CF5 = (("attnglob", 2.0), ("attnwp05", 2.0), ("attn1", 2.0), ("drop1", 1.0), ("r13", 1.0)); ON = [(f"q24f_{c}_s{s}", w) for s in range(8) for c, w in CF5]; WS = sum(w for _, w in ON)
lg = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6))); sg = lambda x: 1 / (1 + np.exp(-x)); tmp = lambda p: sg(lg(p) / TE)
WIN = [w for w in G.WID if G.META[w]["cond"] == "queue_ongoing" and (bool(m3v2.is_holdout(np.array([G.META[w]["day"]]))[0]) == (RUN == "read"))]


def frame(wins, parts):
    PL = [pickle.load(open(W + f"probs_m3v3__tfm4_s{s}.pkl", "rb")) for s in range(5)]; PX = [pickle.load(open(W + f"probs_m3v3cap__x3a_s{s}_fold2030-12.pkl", "rb")) for s in (0, 1)]; PQ = {t: pickle.load(open(W + f"probs_m3v3__{t}.pkl", "rb")) for t, _ in ON}; SD = {}
    for s in (0, 1):
        for f in glob.glob(W + f"x1_arms/{parts}/sd__c24f_attnglob_s{s}__*.pkl"):
            for key, v in pickle.load(open(f, "rb")).items():
                SD.setdefault(key, []).append(v)
    CC = ("A", "bundle", "B", "Bc"); cols = {c: {} for c in CC}
    for w in wins:
        m = G.META[w]; i = G.POS[w]; Xw = np.asarray(G.X[G.OFF[i]:G.OFF[i + 1]]); pl = g.FS.lgb_ongoing(Xw); pt = sum(d[w] for d in PL) / 5; px = (PX[0][w] + PX[1][w]) / 2; pq = sum(wt * PQ[t][w] for t, wt in ON) / WS; lq = np.asarray(Xw[:, G.CI], float) == 1
        P, _, q = m3v2._panel(m["panel"]); t = m["t"]; d0 = t - (t % 288) + 60; _, cone, _ = g.g55.parts(w); nr = g132.nears(lq, P.n)[3]; sc = dict.fromkeys(CC, 0.0)
        for k, pr in m3v2.kdist(m["day"]).items():
            hid = V._hidden_today(m, k); rr = [r for r in range(d0, t) if r not in hid]; ev_ = q[rr].any(axis=0) if rr else np.zeros(P.n, bool); s2, _ = m3v2.seen(m["panel"], t, k, -18, 12); ex = G.dil(ev_ | s2, 2); s, nd = m3v2.seen(m["panel"], t, k, -18, 12); sw = np.where(G.dil(s, 1), 1.0, 0.85)[None, :] if nd else 1.0; cf = np.where(ex[None, :], 1.0, np.where(cone < 0.999, 0.5, 1.0)); x1 = g.X1[(w, k)]; xs = sum(SD[(w, k)]) / 2
            dec = lambda pb: g132.gate(best_set(pb.ravel(), np.ones(pb.size, bool)).reshape(pb.shape), ex, nr)
            sc["A"] += pr * G.iou(dec((0.8 * (0.8 * x1 + 0.2 * pt) + 0.2 * pl) * cf * sw), m); sc["bundle"] += pr * G.iou(dec(tmp((0.8 * ((1 - X3) * (0.8 * xs + 0.2 * pt) + X3 * px) + 0.2 * pl) * cf * sw)), m)
            sc["B"] += pr * G.iou(dec(pq * sw), m); sc["Bc"] += pr * G.iou(dec(tmp(((1 - X3) * pq + X3 * px) * sw)), m)
        for c in CC:
            cols[c][w] = sc[c]
    return V.frame(cols).dropna(subset=["A"]).reset_index(drop=True)


def train_B():
    h = module("g119", R0 + "review/g119_month_postproc.py"); T, GG = h.T, h.G; O = [pickle.load(open(W + f"oof_q25_nomem_s{s}.pkl", "rb")) for s in (3, 4)]; X = [pickle.load(open(f"C:/tfb_data/kaggle_out/cap1/out/oof_x3a_s{s}.pkl", "rb")) for s in (0, 1)]; rows = []
    for mo in h.MONTHS:
        ks = np.flatnonzero(T.META["is_sel"] & (h.MON == mo) & (T.META["cond"] == "queue_ongoing"))
        for p in sorted({str(T.META["panel"][k]) for k in ks}):
            P = h.load(p, "train"); key = (mo, p, "mask"); h.CTX.clear(); h.sim_blank.cache_clear(); h.CTX[key] = h.Ctx(key, P, np.where(P.blank, np.nan, P.V), P.V); c_ = h.CTX[key]
            for k in [k for k in ks if str(T.META["panel"][k]) == p and k in X[0] and k in X[1]]:
                t = int(T.META["t"][k]); lab, el = T.LAB[k]; y, e = lab.astype(bool), el.astype(bool); Xw = np.asarray(h.XN[h.OFFN[k]:h.OFFN[k + 1]]); lq = np.asarray(Xw[:, GG.CI], float) == 1; p_ = (O[0][k] + O[1][k]) / 2; px = (X[0][k] + X[1][k]) / 2; dom = int(P.times[t][8:10]) - 1; nr = g132.nears(lq, P.n)[3]; r = dict(w=int(k), month=mo, panel=p, day=dom, B=0.0, Bc=0.0)
                for kk, pr in m3v2.kdist(dom, cap=3).items():
                    ev_ = h.today_ev(c_, t, kk); s2, _ = h.seen(c_, t, kk, -18, 12); ex = GG.dil(ev_ | s2, 2); s, nd = h.seen(c_, t, kk, -18, 12); sw = np.where(GG.dil(s, 1), 1.0, 0.85)[None, :] if nd else 1.0
                    r["B"] += pr * h.iou(g132.gate(h.decode(p_ * sw), ex, nr), y, e, "queue_ongoing"); r["Bc"] += pr * h.iou(g132.gate(h.decode(tmp(((1 - X3) * p_ + X3 * px) * sw)), ex, nr), y, e, "queue_ongoing")
                rows.append(r)
            h.load.cache_clear()
    return pd.DataFrame(rows)


def boot(d, cl, n_boot=2000):
    ids = pd.factorize(cl)[0]; M = np.bincount(ids, d); N = np.bincount(ids); Rm = np.random.default_rng(0).multinomial(len(M), np.full(len(M), 1 / len(M)), size=n_boot); return float(((Rm @ M) / (Rm @ N)).std())


if RUN == "dev":
    D = frame(WIN, "parts"); D.to_csv(W + "g175_dev.csv", index=False); r = pd.read_csv(W + "g171_m3.csv").set_index("w").loc[D.w]; print(f"{len(D)} windows | identity A, bundle against g171: {np.abs(r.A.to_numpy() - D.A.to_numpy()).max():.2e}, {np.abs(r.bundle.to_numpy() - D.bundle.to_numpy()).max():.2e} | level B {D.B.mean():.4f}", flush=True)
    for a, b, nm in (("A", "bundle", "A side"), ("B", "Bc", "B side")):
        s = m3v2.summary(D, a, b, "private"); sm = m3v2.summary(D, a, b, "validation", n_boot=1000, n_draw=200); z = s["t2_weighted"] / s["se"]; print(f"{nm}: April composition {s['t2_weighted']:+.4f} (SE {s['se']:.4f}, z {z:+.2f}, /1.25 {z / 1.25:+.2f}) | March composition {sm['t2_weighted']:+.4f} | raw {(D[b] - D[a]).mean():+.4f}", flush=True)
    Df = train_B(); Df.to_csv(W + "g175_train_B.csv", index=False)
    for mo, d in Df.groupby("month"):
        print(mo, len(d), f"B {d.B.mean():.4f} Bc - B {(d.Bc - d.B).mean():+.4f} (SE {boot((d.Bc - d.B).to_numpy(), d.panel + '_' + d.day.astype(str)):.4f})", flush=True)
    sys.exit()
D = frame(WIN, "parts_holdout"); D.to_csv(W + "g175_hold.csv", index=False); Dv = pd.read_csv(W + "g175_dev.csv")
for a, b, nm in (("A", "bundle", "A side"), ("B", "Bc", "B side")):
    s = m3v2.summary(D, a, b, "private"); d = m3v2.summary(Dv, a, b, "private"); w1, w2 = 1 / d["se"] ** 2, 1 / s["se"] ** 2; mg = (w1 * d["t2_weighted"] + w2 * s["t2_weighted"]) / (w1 + w2); se = (w1 + w2) ** -0.5; print(f"{nm}: hold-out {len(D)} windows {s['t2_weighted']:+.4f} (SE {s['se']:.4f}) | dev {d['t2_weighted']:+.4f} (SE {d['se']:.4f}) | merged {mg:+.4f} (SE {se:.4f}, z {mg / se:+.2f}, /1.25 {mg / se / 1.25:+.2f}) | line /1.25 >= +2: {mg / se / 1.25 >= 2}")
