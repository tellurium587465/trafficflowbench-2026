"""G227 (11/03, ledger #198): a minimum number of selected cells for the onset set (requires_queue_in_horizon as a prior). best_set takes
the top k of its expected-IoU curve; here k >= KMIN (1 = production, 2, 3), onset windows only, everything else unchanged.
  m3 : M3 v3 DEV onset, final A's onset flow as G120 / G108 column A (q24f 8 seeds CF5 + physics onset with adaptive a, evidence x0.25,
       k expectation, simulated blanks); identity: KMIN 1 == G108 column A. April composition (m3v2.summary), z/1.25.
  jf : 2031-01 / 2031-02 selector onset windows through G119's onset flow PE (OOF q25_nomem s3/s4, LOMO physics model and rarity),
       visibility "full" (unmasked history, as the official windows) primary and "mask" beside it.
usage: TFB_M3VER=v3 py g227_onset_mink.py m3 | jf"""
import os, sys, importlib.util
os.environ.setdefault("TFB_M3VER", "v3")
import numpy as np, pandas as pd
MODE = sys.argv[1]; R = r"C:/Users/user/tfb/"; KS = (1, 2, 3)


def module(nm, path, argv):
    a = sys.argv; sys.argv = argv; spec = importlib.util.spec_from_file_location(nm, path); mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); sys.argv = a
    return mod


def mk(kmin):
    def best_set(p, elig):
        pe = np.where(elig, p, 0.0); order = np.argsort(-pe); cs = np.cumsum(pe[order]); tot = pe.sum()
        k = np.arange(1, len(pe) + 1); kb = max(int(np.argmax(cs / (k + tot - cs))) + 1, kmin)
        sel = np.zeros(len(p), bool); sel[order[:kb]] = True; return sel
    return best_set


if MODE == "m3":
    import lightgbm as lgb
    g2 = module("g120", R + "review/g120_onset_gate.py", ["g120", "run"]); G, V, m3v2, PO = g2.G, g2.V, g2.m3v2, g2.PO
    PN = g2.nn_blend(); phys = lgb.Booster(model_file=G.W + "physon_base.txt"); F_ = PO.rarity_table(); ev = PO.month_events("validation", F_); rows = []
    for w in g2.DEV:
        m = G.META[w]; i = G.POS[w]; Xw = np.asarray(G.X[G.OFF[i]:G.OFF[i + 1]]); ph = phys.predict(PO.feats(Xw)); a_ph = PO.adaptive_a(ev, m["day"])
        r = dict(w=w, cond="queue_onset", panel=m["panel"], day=m["day"], hour=(m["t"] % 288) // 12, **{f"k{q}": 0.0 for q in KS})
        for k, pr in m3v2.kdist(m["day"]).items():
            p = PN[w].copy(); p[5] = (1 - a_ph) * p[5] + a_ph * ph
            for q in KS:
                V.best_set = mk(q); r[f"k{q}"] += pr * G.iou(V.onset_pred(p, m, "cur", k), m)
        rows.append(r)
    D = pd.DataFrame(rows); ref = pd.read_csv(G.W + "g108_rectangles.csv").set_index("w")["A"]
    print(f"{len(D)} dev onset windows | identity KMIN 1 vs G108 column A max |d| {float(np.max(np.abs(D.k1 - ref.loc[D.w].to_numpy()))):.2e}", flush=True)
    for q in KS[1:]:
        s = m3v2.summary(D, "k1", f"k{q}", "private"); z = s["t2_weighted"] / s["se"]; h = [float((D[f"k{q}"] - D.k1)[D.day % 2 == o].mean()) for o in (1, 0)]
        print(f"KMIN {q}: April composition {s['t2_weighted']:+.4f} (SE {s['se']:.4f}, z {z:+.2f}, z/1.25 {z / 1.25:+.2f}) | raw {float((D[f'k{q}'] - D.k1).mean()):+.4f} | odd / even days {h[0]:+.4f} / {h[1]:+.4f} | windows changed {int((np.abs(D[f'k{q}'] - D.k1) > 1e-12).sum())}", flush=True)
    D.to_csv(G.W + "g227_m3.csv", index=False); sys.exit()

g = module("g119", R + "review/g119_month_postproc.py", ["g119", "run"])
sw = g.score_window


def score_window(c, p, t, cond, y, e, lq, dom, length, ph=None, a=0.0):
    if cond != "queue_onset":
        return {}
    out = {}
    for k, pr in g.m3v2.kdist(dom, cap=3).items():
        pp = p.copy(); pp[5] = (1 - a) * p[5] + a * ph
        for q in KS:
            g.best_set = mk(q); out[f"k{q}"] = out.get(f"k{q}", 0.0) + pr * g.iou(g.onset_pred(c, pp, t, k, "cur"), y, e, cond)
    g.best_set = mk(1); return out


g.score_window = score_window
D = g.train_months(); D = D[D.cond == "queue_onset"].reset_index(drop=True); D.to_csv(g.W + "g227_jf.csv", index=False)


def boot(d, cl, n_boot=2000, seed=0):
    rng = np.random.default_rng(seed); ids = pd.factorize(cl)[0]; M = np.bincount(ids, d); N = np.bincount(ids)
    Rm = rng.multinomial(len(M), np.full(len(M), 1.0 / len(M)), size=n_boot); return float(((Rm @ M) / (Rm @ N)).std())


for vis in ("full", "mask"):
    E = D[D.vis == vis]; cl = (E.month + "_" + E.panel + "_" + E.day.astype(str)).to_numpy()
    for q in KS[1:]:
        d = (E[f"k{q}"] - E.k1).to_numpy(); se = boot(d, cl)
        print(f"{vis}: KMIN {q} - 1 over {len(E)} onset windows: merged {d.mean():+.4f} (SE {se:.4f}, z {d.mean() / max(se, 1e-12):+.2f}) | "
              + " | ".join(f"{mo} {d[(E.month == mo).to_numpy()].mean():+.4f}" for mo in g.MONTHS), flush=True)
