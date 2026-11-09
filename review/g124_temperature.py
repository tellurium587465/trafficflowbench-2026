"""G124 (10/30, ledger #78, strategy J18): a temperature on the probabilities given to best_set,
p' = sigmoid(logit(p) / Te), Te in {0.7, 0.85, 1.2, 1.5} (Te = 1 through the same function = the identity).
  m3og  : M3 v3 DEV ongoing, final A's flow (G121 path: X1 0.8 + LightGBM 0.2, cone, seen x0.85 -> Te -> best_set -> far gate)
  m3on  : M3 v3 DEV onset, final A's flow (G120 path: q24f CF5 + physics adaptive a, evidence x0.25 -> Te -> best_set)
  train : 2031-01 / 2031-02 selector-like windows (G119 path, mask): raw = Te -> best_set (both conditions); ongoing SCG (record)
usage: py g124_temperature.py [m3og | m3on | train]"""
import os, sys, time, pickle, importlib.util
from collections import defaultdict
MODE = (sys.argv[1:] or ["m3og"])[0]; TES = (1.0, 0.7, 0.85, 1.2, 1.5)
import numpy as np, pandas as pd
TM = time.perf_counter; T00 = TM(); R = r"C:/Users/user/tfb/"


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path); mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


def temp(p, te):
    q = np.clip(p, 1e-7, 1 - 1e-7); return 1.0 / (1.0 + np.exp(-(np.log(q) - np.log1p(-q)) / te))


def boot(d, cl, n_boot=2000, seed=0):
    rng = np.random.default_rng(seed); ids = pd.factorize(cl)[0]; M = np.bincount(ids, d); N = np.bincount(ids)
    Rm = rng.multinomial(len(M), np.full(len(M), 1.0 / len(M)), size=n_boot); return float(((Rm @ M) / (Rm @ N)).std())


def m3og():
    g = module("g121", R + "review/g121_ongoing_oracle.py"); G, V, m3v2, best_set = g.G, g.V, g.m3v2, g.best_set; rows = []
    for w in g.DEV:
        m = G.META[w]; i = G.POS[w]; pl = g.FS.lgb_ongoing(np.asarray(G.X[G.OFF[i]:G.OFF[i + 1]])); r = defaultdict(float, w=w, panel=m["panel"], day=m["day"])
        for k, pr in m3v2.kdist(m["day"]).items():
            p = (0.8 * g.X1[(w, k)] + 0.2 * pl) * g.cone_f(m, k); s, nd = m3v2.seen(m["panel"], m["t"], k, -18, 12)
            pb = p * np.where(G.dil(s, 1), 1.0, 0.85)[None, :] if nd else p.copy()
            for te in TES:
                pt = temp(pb, te); sel = V.far_gate(best_set(pt.ravel(), np.ones(pt.size, bool)).reshape(pt.shape), m, k); r[f"t{te}"] += pr * G.iou(sel, m)
        rows.append(dict(r))
    return pd.DataFrame(rows), g.W, [("t", "M3 dev ongoing")]


def m3on():
    g = module("g120", R + "review/g120_onset_gate.py"); G, m3v2, PO, best_set = g.G, g.m3v2, g.PO, g.best_set
    PN = g.nn_blend(); phys = g.lgb.Booster(model_file=g.W + "physon_base.txt"); ev = PO.month_events("validation", PO.rarity_table()); rows = []
    for w in g.DEV:
        m = G.META[w]; i = G.POS[w]; Xw = np.asarray(G.X[G.OFF[i]:G.OFF[i + 1]]); ph = phys.predict(PO.feats(Xw)); a = PO.adaptive_a(ev, m["day"])
        r = defaultdict(float, w=w, panel=m["panel"], day=m["day"])
        for k, pr in m3v2.kdist(m["day"]).items():
            pb = PN[w].copy(); pb[5] = (1 - a) * pb[5] + a * ph; s, nd = m3v2.seen(m["panel"], m["t"], k, -6, 12)     # V.onset_pred "cur"
            if nd:
                pb[-1] = pb[-1] * np.where(G.dil(s, 2), 1.0, 0.25)
            for te in TES:
                pt = temp(pb, te); r[f"t{te}"] += pr * G.iou(best_set(pt.ravel(), np.ones(pt.size, bool)).reshape(pt.shape), m)
        rows.append(dict(r))
    return pd.DataFrame(rows), g.W, [("t", "M3 dev onset")]


def train():
    g = module("g119", R + "review/g119_month_postproc.py"); T, G, m3v2 = g.T, g.G, g.m3v2
    OOF = [pickle.load(open(g.W + f"oof_q25_nomem_s{s}.pkl", "rb")) for s in (3, 4)]; rows = []
    for mo in g.MONTHS:
        ks = np.flatnonzero(T.META["is_sel"] & (g.MON == mo))
        for p in sorted({str(T.META["panel"][k]) for k in ks}):
            P = g.load(p, "train"); key = (mo, p, "mask"); g.CTX.clear(); g.sim_blank.cache_clear()
            g.CTX[key] = g.Ctx(key, P, np.where(P.blank, np.nan, P.V), P.V); c = g.CTX[key]
            for k in [k for k in ks if str(T.META["panel"][k]) == p]:
                t = int(T.META["t"][k]); cond = str(T.META["cond"][k]); dom = int(P.times[t][8:10]) - 1; lab, el = T.LAB[k]; y, e = lab.astype(bool), el.astype(bool)
                p_ = (OOF[0][k] + OOF[1][k]) / 2; r = defaultdict(float, w=int(k), month=mo, cond=cond, panel=p, day=dom)
                for te in TES:
                    r[f"raw_t{te}"] = g.iou(g.decode(temp(p_, te)), y, e, cond)
                if cond == "queue_ongoing":
                    Xw = np.asarray(g.XN[g.OFFN[k]:g.OFFN[k + 1]]); lq = np.asarray(Xw[:, G.CI], float) == 1; cone = g.cone_parts(c, t, lq, P.length)
                    for kk, pr in m3v2.kdist(dom, cap=3).items():
                        pc = p_ * g.cone_f(c, cone, t, kk); s_, nd = g.seen(c, t, kk, -18, 12)
                        pb = pc * np.where(G.dil(s_, 1), 1.0, 0.85)[None, :] if nd else pc.copy()
                        for te in TES:
                            r[f"SCG_t{te}"] += pr * g.iou(g.far_gate(c, g.decode(temp(pb, te)), t, kk, lq), y, e, cond)
                rows.append(dict(r))
            g.load.cache_clear()
    D = pd.DataFrame(rows); ref = pd.read_csv(g.W + "g119_month_postproc.csv", dtype={"w": str})
    ref = ref[ref.vis == "mask"].set_index(["month", "w"]); D["raw"] = [ref.loc[(r.month, str(r.w)), "raw"] for r in D.itertuples()]
    D["SCG"] = [ref.loc[(r.month, str(r.w)), "SCG"] if r.cond == "queue_ongoing" else np.nan for r in D.itertuples()]
    return D, g.W, None


if __name__ == "__main__":
    D, W, spec = {"m3og": m3og, "m3on": m3on, "train": train}[MODE]()
    print(f"{MODE}: {len(D)} windows in {TM() - T00:.0f}s")
    if MODE != "train":
        ref = pd.read_csv(W + "g108_rectangles.csv").set_index("w")["A"]; d0 = float(np.max(np.abs(D["t1.0"] - ref.loc[D.w].to_numpy())))
        print(f"identity: Te = 1 vs G108 column A max |d| {d0:.2e}")
        groups = [(spec[0][1], D, "t1.0", "t")]
    else:
        d0 = max(float(np.nanmax(np.abs(D["raw_t1.0"] - D.raw))), float(np.nanmax(np.abs(D["SCG_t1.0"] - D.SCG))))
        print(f"identity: Te = 1 vs g119 raw / SCG max |d| {d0:.2e}")
        groups = [(f"{mo} {c.split('_')[1]} {col}", D[(D.month == mo) & (D.cond == c)], f"{col}_t1.0", f"{col}_t")
                  for mo in ("2031-01", "2031-02") for c in ("queue_onset", "queue_ongoing") for col in (("raw",) if c == "queue_onset" else ("raw", "SCG"))]
    if d0 > 1e-12:
        print("IDENTITY FAILED: stop"); sys.exit(1)
    D.to_csv(R + f"work/t2/g124_temperature_{MODE}.csv", index=False)
    for nm, d, base, pre in groups:
        cl = d.panel.astype(str) + "_" + d.day.astype(str)
        print(f"{nm}: windows {len(d)} | base {d[base].mean():.4f} | " + " | ".join(
            f"Te {te}: {(d[f'{pre}{te}'] - d[base]).mean():+.4f} (z {(d[f'{pre}{te}'] - d[base]).mean() / max(boot((d[f'{pre}{te}'] - d[base]).to_numpy(), cl), 1e-12):+.1f})"
            for te in TES[1:]))
