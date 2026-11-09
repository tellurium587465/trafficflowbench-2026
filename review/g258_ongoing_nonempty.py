"""G258 (11/04, ledger #219, the strategy's 05:50 request, (N)(ii) on ongoing): the official windows are drawn on a non-empty truth
(requires_queue_in_horizon), so the ongoing window's cells rescaled p' = p / (1 - prod(1 - p)) over the whole horizon before the decode:
nza = best_set on the rescaled, exnza = DEX on the rescaled, ex = DEX alone (as #216). Instruments as G254 (jf = G229's January /
February 316 windows; dev = G212's dev frame, B10's flow, all and the top coverage third). Also the calibration sum(p) / sum(y) per window.
usage: TFB_X3=x3af TFB_M3VER=v3 py -B -X utf8 g258_ongoing_nonempty.py jf | dev"""
import os, sys, pickle, importlib.util
os.environ.setdefault("TFB_M3VER", "v3")
import numpy as np, pandas as pd
sys.path.insert(0, r"C:/Users/user/tfb/review"); import dex as _dx
NZA = lambda pb: np.clip(np.asarray(pb, float) / max(1 - np.prod(1 - np.clip(np.asarray(pb, float), 0, 1 - 1e-6)), 1e-6), 0.0, 1 - 1e-6)
R = r"C:/Users/user/tfb/"; MODE = sys.argv[1]
if MODE == "jf":
    a_ = sys.argv; sys.argv = ["g119", "run"]; spec = importlib.util.spec_from_file_location("g119", R + "review/g119_month_postproc.py"); g = importlib.util.module_from_spec(spec); spec.loader.exec_module(g); sys.argv = a_
    T, MON = g.T, g.MON; KS = [k for k in np.flatnonzero(T.META["is_sel"] & np.isin(MON, g.MONTHS)) if str(T.META["cond"][k]) == "queue_ongoing"]
    PR = [pickle.load(open(g.W + f"probs_jfmh0__capB0_s{s}_fold2030-12.pkl", "rb")) for s in range(4)]; dec0 = g.decode; rows = []
    for mo in g.MONTHS:
        for p in sorted({str(T.META["panel"][k]) for k in KS if MON[k] == mo}):
            P = g.load(p, "train"); key = (mo, p, "mask"); g.CTX.clear(); g.sim_blank.cache_clear(); g.CTX[key] = g.Ctx(key, P, np.where(P.blank, np.nan, P.V), P.V); c = g.CTX[key]
            for k in [k for k in KS if MON[k] == mo and str(T.META["panel"][k]) == p]:
                t = int(T.META["t"][k]); dom = int(P.times[t][8:10]) - 1; lab, el = T.LAB[k]; y, e = lab.astype(bool), el.astype(bool)
                fin = np.isfinite(P.V[t + 1:t + 7]); yF = fin & (np.nan_to_num(P.V[t + 1:t + 7], nan=1e9) <= P.vcut[None, :])
                lq = np.asarray(g.XN[g.OFFN[k]:g.OFFN[k + 1]][:, g.G.CI], float) == 1; ot = c.obs[t].astype(bool); lq2 = np.where(ot, c.q[t].astype(bool), lq)
                p_ = sum(d[str(k)] for d in PR) / len(PR); cone = g.cone_parts(c, t, lq2, P.length); r = dict(w=int(k), month=mo, panel=p, day=dom)
                r["sp"] = float(p_.sum()); r["sy"] = float((y & e).sum())
                for nm, dfun in (("base", dec0), ("ex", _dx.DEX), ("nza", lambda pb: dec0(NZA(pb))), ("exnza", lambda pb: _dx.DEX(NZA(pb)))):
                    g.decode = dfun; sl = sf = 0.0
                    for kk, pr in g.m3v2.kdist(dom, cap=3).items():
                        sel = g.ongoing_pred(c, p_ * g.cone_f(c, cone, t, kk), t, kk, True, lq2); sl += pr * g.iou(sel, y, e, "queue_ongoing"); sf += pr * g.iou(sel, yF, fin, "queue_ongoing")
                    r[f"L_{nm}"] = sl; r[f"F_{nm}"] = sf
                g.decode = dec0; rows.append(r)
            g.load.cache_clear()
    D = pd.DataFrame(rows); D.to_csv(g.W + "g258_jf.csv", index=False); ref = pd.read_csv(g.W + "g229_jf.csv").set_index("w").trow.reindex(D.w).to_numpy()
    print(f"{len(D)} windows | identity L_base vs G229 trow max |d| {np.abs(ref - D.L_base.to_numpy()).max():.2e}", flush=True)
    cl = (D.month.astype(str) + "_" + D.panel + "_" + D.day.astype(str)).to_numpy()
    def boot(d, cl, n=2000):
        ids = pd.factorize(cl)[0]; M_ = np.bincount(ids, d); N_ = np.bincount(ids); Rm = np.random.default_rng(0).multinomial(len(M_), np.full(len(M_), 1 / len(M_)), size=n); return float(((Rm @ M_) / (Rm @ N_)).std())
    print(f"calibration sum(p) / sum(y) over the windows: {D.sp.sum() / D.sy.sum():.3f} | per-window median {np.median(D.sp / np.maximum(D.sy, 1)):.3f}", flush=True)
    for lab, v in (("F", "ex"), ("F", "nza"), ("F", "exnza"), ("L", "nza")):
        for mo in ("pool",):
            m_ = np.ones(len(D), bool); d = (D[f"{lab}_{v}"] - D[f"{lab}_base"]).to_numpy()[m_]
            print(f"{v} - base, {lab}, {mo}: {d.mean():+.4f} (SE {boot(d, cl[m_]):.4f}, z {d.mean() / max(boot(d, cl[m_]), 1e-12):+.2f}) | windows changed {int((np.abs(d) > 1e-12).sum())}", flush=True)
    sys.exit()
assert MODE == "dev"
R_ = R + "review/"; END = 'exec(compile(s, "g175a", "exec"))'; src = open(R_ + "g212_a10_read.py", encoding="utf-8").read(); assert src.count(END) == 1
exec(compile(src.replace(END, ""), "g212", "exec"))
F_ = '(0.8 * (0.8 * ((1 - X3) * (0.8 * xs + 0.2 * pt) + X3 * px) + 0.2 * pu) + 0.2 * pl) * cf * sw'
sub('sc["a12"] += pr * G.iou(dec(tmp(' + F_ + ')), m)', 'sc["a12"] += pr * G.iou(dec(tmp(' + F_ + ')), m); sc["a12x"] += pr * G.iou(g132.gate(_dx.DEX(tmp(' + F_ + ')), ex, nr), m)')
sub('sc["b10"] += pr * G.iou(raw(tmp(0.8 * ((1 - X3) * pq + X3 * px) + 0.2 * (PU[0][w] + PU[1][w]) / 2)), m);',
    'sc["b10"] += pr * G.iou(raw(tmp(0.8 * ((1 - X3) * pq + X3 * px) + 0.2 * (PU[0][w] + PU[1][w]) / 2)), m); sc["b10x"] += pr * G.iou(_dx.DEX(tmp(0.8 * ((1 - X3) * pq + X3 * px) + 0.2 * (PU[0][w] + PU[1][w]) / 2)), m); pB_ = tmp(0.8 * ((1 - X3) * pq + X3 * px) + 0.2 * (PU[0][w] + PU[1][w]) / 2); sc["b10n"] += pr * G.iou(raw(NZA(pB_)), m); sc["b10xn"] += pr * G.iou(_dx.DEX(NZA(pB_)), m); CAL.append((float(pB_.sum()), float((m["y"].T.astype(bool) & m["m"].T.astype(bool)).sum())));')
sub('"a12", "b10"); cols', '"a12", "b10", "a12x", "b10x", "b10n", "b10xn"); cols')
sub('D.to_csv(W + "g212_dev.csv"', 'D.to_csv(W + "g258_dev.csv"')
sub('    Df = train_B(); Df.to_csv(W + "g212_train.csv", index=False)\n',
    '    cov_ = COV(); D["cov"] = D.w.map(cov_); top = D[D["cov"] > D["cov"].quantile(2 / 3)].reset_index(drop=True)\n'
    '    print(f"calibration (B10 flow) sum(p) / sum(y): {sum(c[0] for c in CAL) / sum(c[1] for c in CAL):.3f}", flush=True)\n'
    '    for a_, b_ in (("b10", "b10x"), ("b10", "b10n"), ("b10", "b10xn")):\n'
    '        for nm_, d_ in (("all", D), ("top coverage third", top)):\n'
    '            s_ = m3v2.summary(d_, a_, b_, "private", n_boot=2000, n_draw=500); print(f"{b_} - {a_} ({nm_}, {len(d_)}): {s_[\'t2_weighted\']:+.4f} (SE {s_[\'se\']:.4f}, z {s_[\'t2_weighted\'] / max(s_[\'se\'], 1e-12):+.2f}) | windows changed {int((~np.isclose(d_[b_], d_[a_])).sum())}", flush=True)\n'
    '    sys.exit()\n')
s = "CAL = []\n" + s
exec(compile(s, "g175x", "exec"))
