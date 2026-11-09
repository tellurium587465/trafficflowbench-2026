"""G171 (11/01, ledger #121, strategy T2_DEEP_REVIEW 8-1 A2). (1) The early adopted components on one scale: M3 dev, judge, A minus A without it (g156, g170), z, z/1.25 for models, and the late line. (2) One bundle of the cheap positive points,
k = 1: same-day X1 (#101, arm sd) + X3a 0.2 + ongoing temperature 1.2, on final A's March flow: 0.8 (0.8 ((0.8 X1 + 0.2 tfm)) + 0.2 X3a) + 0.2 LightGBM -> cone -> seen x0.85 -> temperature -> best_set -> far gate. M3 dev ongoing 356 windows.
January and February ongoing: the same without X1 (no sd arm exists there): 0.8 OOF + 0.2 X3a OOF -> cone -> seen -> temperature -> best_set -> far gate, against G119's SCG. Identity: A against g132's base cell, the January and February base against G119's SCG.
usage: TFB_M3VER=v3 py -B -X utf8 g171_bundle.py [dry | run]"""
import os, sys, glob, pickle, importlib.util
os.environ.setdefault("TFB_M3VER", "v3"); sys.dont_write_bytecode = True; RUN = (sys.argv[1:] or ["run"])[0]; R0 = r"C:/Users/user/tfb/"
import numpy as np, pandas as pd


def module(nm, path):
    spec = importlib.util.spec_from_file_location(nm, path); mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


sys.argv = [sys.argv[0], "run"]; g = module("g121", R0 + "review/g121_ongoing_oracle.py"); g132 = module("g132", R0 + "review/g132_p3_grid.py")
G, V, m3v2, best_set = g.G, g.V, g.m3v2, g.best_set; W = g.W; TE = 1.2; X3 = 0.2
lg = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6))); sg = lambda x: 1 / (1 + np.exp(-x)); tmp = lambda p: sg(lg(p) / TE)


def m3(dev):
    PL = [pickle.load(open(W + f"probs_m3v3__tfm4_s{s}.pkl", "rb")) for s in range(5)]; PX = [pickle.load(open(W + f"probs_m3v3cap__x3a_s{s}_fold2030-12.pkl", "rb")) for s in (0, 1)]; SD = {}
    for s in (0, 1):
        for f in glob.glob(W + f"x1_arms/parts/sd__c24f_attnglob_s{s}__*.pkl"):
            for key, v in pickle.load(open(f, "rb")).items():
                SD.setdefault(key, []).append(v)
    print("sd entries", len(SD), "with both seeds", sum(len(v) == 2 for v in SD.values()), flush=True); CC = ("A", "sd", "x3", "te", "bundle"); cols = {c: {} for c in CC}
    for w in dev:
        m = G.META[w]; i = G.POS[w]; Xw = np.asarray(G.X[G.OFF[i]:G.OFF[i + 1]]); pl = g.FS.lgb_ongoing(Xw); pt = sum(d[w] for d in PL) / 5; px = (PX[0][w] + PX[1][w]) / 2; lq = np.asarray(Xw[:, G.CI], float) == 1
        P, _, q = m3v2._panel(m["panel"]); t = m["t"]; d0 = t - (t % 288) + 60; _, cone, _ = g.g55.parts(w); nr = g132.nears(lq, P.n)[3]; sc = dict.fromkeys(CC, 0.0)
        for k, pr in m3v2.kdist(m["day"]).items():
            hid = V._hidden_today(m, k); rr = [r for r in range(d0, t) if r not in hid]; ev_ = q[rr].any(axis=0) if rr else np.zeros(P.n, bool); s2, _ = m3v2.seen(m["panel"], t, k, -18, 12); ex = G.dil(ev_ | s2, 2); s, nd = m3v2.seen(m["panel"], t, k, -18, 12); sw = np.where(G.dil(s, 1), 1.0, 0.85)[None, :] if nd else 1.0; cf = np.where(ex[None, :], 1.0, np.where(cone < 0.999, 0.5, 1.0)); x1 = g.X1[(w, k)]; xs = sum(SD[(w, k)]) / 2

            def dec(x, wx, te):
                pb = (0.8 * ((1 - wx) * (0.8 * x + 0.2 * pt) + wx * px) + 0.2 * pl) * cf * sw; pb = tmp(pb) if te else pb; return g132.gate(best_set(pb.ravel(), np.ones(pb.size, bool)).reshape(pb.shape), ex, nr)
            for c, a in (("A", (x1, 0.0, False)), ("sd", (xs, 0.0, False)), ("x3", (x1, X3, False)), ("te", (x1, 0.0, True)), ("bundle", (xs, X3, True))):
                sc[c] += pr * G.iou(dec(*a), m)
        for c in CC:
            cols[c][w] = sc[c]
    return V.frame(cols).dropna(subset=["A"]).reset_index(drop=True)


def train(limit=None):
    h = module("g119", R0 + "review/g119_month_postproc.py"); T, GG = h.T, h.G; O = [pickle.load(open(W + f"oof_q25_nomem_s{s}.pkl", "rb")) for s in (3, 4)]; X = [pickle.load(open(f"C:/tfb_data/kaggle_out/cap1/out/oof_x3a_s{s}.pkl", "rb")) for s in (0, 1)]; rows = []
    for mo in h.MONTHS:
        ks = np.flatnonzero(T.META["is_sel"] & (h.MON == mo) & (T.META["cond"] == "queue_ongoing"))
        for p in sorted({str(T.META["panel"][k]) for k in ks}):
            P = h.load(p, "train"); key = (mo, p, "mask"); h.CTX.clear(); h.sim_blank.cache_clear(); h.CTX[key] = h.Ctx(key, P, np.where(P.blank, np.nan, P.V), P.V); c_ = h.CTX[key]
            for k in [k for k in ks if str(T.META["panel"][k]) == p and k in X[0] and k in X[1]]:
                t = int(T.META["t"][k]); lab, el = T.LAB[k]; y, e = lab.astype(bool), el.astype(bool); Xw = np.asarray(h.XN[h.OFFN[k]:h.OFFN[k + 1]]); lq = np.asarray(Xw[:, GG.CI], float) == 1; p_ = (O[0][k] + O[1][k]) / 2; px = (X[0][k] + X[1][k]) / 2; dom = int(P.times[t][8:10]) - 1; cone = h.cone_parts(c_, t, lq, P.length); nr = g132.nears(lq, P.n)[3]; r = dict(w=int(k), month=mo, panel=p, day=dom, base=0.0, x3=0.0, te=0.0, bundle=0.0)
                for kk, pr in m3v2.kdist(dom, cap=3).items():
                    ev_ = h.today_ev(c_, t, kk); s2, _ = h.seen(c_, t, kk, -18, 12); ex = GG.dil(ev_ | s2, 2); s, nd = h.seen(c_, t, kk, -18, 12); sw = np.where(GG.dil(s, 1), 1.0, 0.85)[None, :] if nd else 1.0; cf = np.where(ex[None, :], 1.0, np.where(cone < 0.999, 0.5, 1.0))
                    for nm, (wx, te) in dict(base=(0.0, False), x3=(X3, False), te=(0.0, True), bundle=(X3, True)).items():
                        pb = ((1 - wx) * p_ + wx * px) * cf * sw; pb = tmp(pb) if te else pb; r[nm] += pr * h.iou(g132.gate(h.decode(pb), ex, nr), y, e, "queue_ongoing")
                rows.append(r)
                if limit and len(rows) >= limit:
                    return pd.DataFrame(rows)
            h.load.cache_clear()
    Df = pd.DataFrame(rows); ref = pd.read_csv(W + "g119_month_postproc.csv", dtype={"w": str}); ref = ref[(ref.vis == "mask") & (ref.cond == "queue_ongoing")].set_index(["month", "w"]).SCG; print("identity base vs G119 SCG", float(np.max(np.abs(Df.base.to_numpy() - np.array([ref.loc[(r.month, str(r.w))] for r in Df.itertuples()])))), len(Df), flush=True); return Df


def boot(d, cl, n_boot=2000):
    ids = pd.factorize(cl)[0]; M = np.bincount(ids, d); N = np.bincount(ids); Rm = np.random.default_rng(0).multinomial(len(M), np.full(len(M), 1 / len(M)), size=n_boot); return float(((Rm @ M) / (Rm @ N)).std())


if __name__ == "__main__":
    dev = g.DEV[:5] if RUN == "dry" else g.DEV; D = m3(dev); ref = pd.read_csv(W + "p3_grid/g132_grid_m3.csv").set_index("w")["l20_t20_c50_D3"]; da = float(np.abs(ref.loc[D.w].to_numpy() - D.A.to_numpy()).max()); print(f"{len(D)} windows | identity A vs g132: {da:.2e}", flush=True)
    if da > 1e-9:
        print("IDENTITY FAILED"); sys.exit(1)
    Df = train(4 if RUN == "dry" else None)
    if RUN == "dry":
        sys.exit()
    D.to_csv(W + "g171_m3.csv", index=False); Df.to_csv(W + "g171_train.csv", index=False)
    for c in ("sd", "x3", "te", "bundle"):
        s = m3v2.summary(D, "A", c, "private"); sm = m3v2.summary(D, "A", c, "validation", n_boot=1000, n_draw=200); z = s["t2_weighted"] / s["se"]; hv = [m3v2.summary(D[D.day % 2 == h_], "A", c, "private", n_boot=300, n_draw=100)["t2_weighted"] for h_ in (1, 0)]
        print(f"M3 dev ongoing {c:7s}: raw {(D[c] - D.A).mean():+.4f} | April composition {s['t2_weighted']:+.4f} (SE {s['se']:.4f}, z {z:+.2f}, /1.25 {z / 1.25:+.2f}) | March composition {sm['t2_weighted']:+.4f} | halves {hv[0]:+.4f} / {hv[1]:+.4f}", flush=True)
    for mo, d in Df.groupby("month"):
        print(mo, len(d), "base", round(d.base.mean(), 4), " | ".join(f"{c} {(d[c] - d.base).mean():+.4f} (SE {boot((d[c] - d.base).to_numpy(), d.panel + '_' + d.day.astype(str)):.4f})" for c in ("x3", "te", "bundle")), flush=True)
    F = pd.read_csv(W + "g156_m3_frame.csv"); O = pd.read_csv(W + "g170_m3.csv"); print("(1) early components on one scale: A minus it, April composition, judge's SE")
    for nm, Dx, a, b, md in (("far gate", F, "no_gate", "A", False), ("X1", F, "no_x1", "A", True), ("cone", F, "no_cone", "A", False), ("LightGBM 0.2", F, "no_lgb", "A", True), ("tfm 0.2", F, "no_tfm", "A", True), ("evidence x0.25", O, "B1.0", "base", False), ("physics onset", O, "a0", "base", True)):
        s = m3v2.summary(Dx, a, b, "private"); z = s["t2_weighted"] / s["se"]; zm = z / 1.25 if md else z; e22 = float(np.sqrt(2 * np.log(22)) - 0.5); print(f"  {nm:15s} {s['t2_weighted']:+.4f} (SE {s['se']:.4f}) z {z:+.2f} {'/1.25 ' + format(zm, '+.2f') if md else ''} | early rule (point > 0, no harm): pass | late line z' - E[max 22] = {zm - e22:+.2f} >= 2: {zm - e22 >= 2}", flush=True)
