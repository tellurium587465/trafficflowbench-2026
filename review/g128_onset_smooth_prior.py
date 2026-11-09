"""G128 (10/30, ledger #82, strategy J22 (b)): the #61 (G114) evidence-rule mix with a smoothed site prior. Where earlier
days of the month exist, the T+30 probabilities x (0.25 + 0.75 eps), eps = (1 - w) e + w fbar, e = the same-month evidence
(+-2 links), fbar = a link within +-2 links whose training-month T+30 frequency averaged over every weekend x 2-hour key
of the panel (rarity_table) is >= 0.1 (#61 used the window's own key). w in (0.25, 0.5, 1); w = 0 = final A.
  m3    : M3 v3 DEV onset (held-out days not read), final A's onset flow (G120 path), base = G108 column A
  train : 2031-01 / 2031-02 selector-like onset windows, G119 path (month-CV OOF q25_nomem s3/s4, mask visibility, physics
          onset / rarity / adaptive a with the month left out), PE = final A's onset form, base = G119 column A
usage: TFB_M3VER=v3 py g128_onset_smooth_prior.py [m3 | train]"""
import os, sys, time, importlib.util
from collections import defaultdict
MODE = (sys.argv[1:] or ["m3"])[0]; WS = (0.0, 0.25, 0.5, 1.0)
os.environ.setdefault("TFB_M3VER", "v3")
import numpy as np, pandas as pd
TM = time.perf_counter; T00 = TM(); R = r"C:/Users/user/tfb/"


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path); mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


def fbar(table, p, dil):
    return dil(np.mean([f for (pp, we, sl), f in table.items() if pp == p], axis=0) >= 0.1, 2).astype(float)


def rule(pb, e, fb, wt):
    out = pb.copy(); out[-1] = out[-1] * (0.25 + 0.75 * ((1 - wt) * e + wt * fb)); return out


def boot(d, cl, n_boot=2000, seed=0):
    rng = np.random.default_rng(seed); ids = pd.factorize(cl)[0]; M = np.bincount(ids, d); N = np.bincount(ids)
    Rm = rng.multinomial(len(M), np.full(len(M), 1.0 / len(M)), size=n_boot); return float(((Rm @ M) / (Rm @ N)).std())


def m3():
    g = module("g120", R + "review/g120_onset_gate.py"); G, m3v2, PO, best_set = g.G, g.m3v2, g.PO, g.best_set
    PN = g.nn_blend(); phys = g.lgb.Booster(model_file=g.W + "physon_base.txt"); RT = PO.rarity_table(); ev = PO.month_events("validation", RT); rows = []
    FB = {p: fbar(RT, p, G.dil) for p in {G.META[w]["panel"] for w in g.DEV}}
    for w in g.DEV:
        m = G.META[w]; i = G.POS[w]; Xw = np.asarray(G.X[G.OFF[i]:G.OFF[i + 1]]); ph = phys.predict(PO.feats(Xw)); a = PO.adaptive_a(ev, m["day"])
        n = m["y"].shape[0]; fb = FB[m["panel"]][:n]; r = defaultdict(float, w=w, panel=m["panel"], day=m["day"])
        for k, pr in m3v2.kdist(m["day"]).items():
            pb = PN[w].copy(); pb[5] = (1 - a) * pb[5] + a * ph; s, nd = m3v2.seen(m["panel"], m["t"], k, -6, 12); e = G.dil(s, 2).astype(float)
            for wt in WS:
                q = rule(pb, e, fb, wt) if nd else pb
                r[f"w{wt}"] += pr * G.iou(best_set(q.ravel(), np.ones(q.size, bool)).reshape(q.shape), m)
        rows.append(dict(r))
    D = pd.DataFrame(rows); ref = pd.read_csv(g.W + "g108_rectangles.csv").set_index("w")["A"]
    return D, float(np.max(np.abs(D["w0.0"] - ref.loc[D.w].to_numpy()))), "G108 column A", g.W


def train():
    g = module("g119", R + "review/g119_month_postproc.py"); T, G, m3v2, PO = g.T, g.G, g.m3v2, g.PO
    F = g.phys_feats(); rows = []
    for mo in g.MONTHS:
        mdl = g.phys_lomo(F, mo); F_ = g.rarity_lomo(mo)
        ks = np.flatnonzero(T.META["is_sel"] & (g.MON == mo) & np.array([str(c) == "queue_onset" for c in T.META["cond"]]))
        for p in sorted({str(T.META["panel"][k]) for k in ks}):
            P = g.load(p, "train"); key = (mo, p, "mask"); g.CTX.clear(); g.sim_blank.cache_clear()
            g.CTX[key] = g.Ctx(key, P, np.where(P.blank, np.nan, P.V), P.V); c = g.CTX[key]; ev = g.month_events(c, p, P, mo, F_); fb = fbar(F_, p, G.dil)
            for k in [k for k in ks if str(T.META["panel"][k]) == p]:
                t = int(T.META["t"][k]); dom = int(P.times[t][8:10]) - 1; lab, el = T.LAB[k]; Xw = np.asarray(g.XN[g.OFFN[k]:g.OFFN[k + 1]])
                p_ = (g.OOF[0][k] + g.OOF[1][k]) / 2 if hasattr(g, "OOF") else None
                if p_ is None:
                    p_ = (OOF[0][k] + OOF[1][k]) / 2
                ph = mdl.predict(PO.feats(Xw)); a = PO.adaptive_a(ev, dom); pp = p_.copy(); pp[5] = (1 - a) * p_[5] + a * ph
                r = defaultdict(float, w=int(k), month=mo, panel=p, day=dom)
                for kk, pr in m3v2.kdist(dom, cap=3).items():
                    s, nd = g.seen(c, t, kk, -6, 12); e = G.dil(s, 2).astype(float)
                    for wt in WS:
                        q = rule(pp, e, fb[:pp.shape[1]], wt) if nd else pp
                        r[f"w{wt}"] += pr * g.iou(g.decode(q), lab.astype(bool), el.astype(bool), "queue_onset")
                rows.append(dict(r))
            g.load.cache_clear()
        print(mo, "done", len(rows), f"{TM() - T00:.0f}s", flush=True)
    D = pd.DataFrame(rows); ref = pd.read_csv(g.W + "g119_month_postproc.csv", dtype={"w": str})
    ref = ref[(ref.vis == "mask") & (ref.cond == "queue_onset")].set_index(["month", "w"])["A"]
    d0 = float(np.max(np.abs(D["w0.0"].to_numpy() - np.array([ref.loc[(r.month, str(r.w))] for r in D.itertuples()]))))
    return D, d0, "G119 column A (onset, mask)", g.W


if __name__ == "__main__":
    if MODE == "train":
        import pickle
        OOF = [pickle.load(open(R + f"work/t2/oof_q25_nomem_s{s}.pkl", "rb")) for s in (3, 4)]
    D, d0, refname, W = {"m3": m3, "train": train}[MODE]()
    print(f"{MODE}: {len(D)} windows in {TM() - T00:.0f}s | identity w = 0 vs {refname} max |d| {d0:.2e}")
    if d0 > 1e-9:
        print("IDENTITY FAILED: stop"); sys.exit(1)
    D.to_csv(W + f"g128_onset_smooth_prior_{MODE}.csv", index=False)
    for nm, d in ([("M3 dev onset", D)] if MODE == "m3" else [(f"{mo} onset PE", D[D.month == mo]) for mo in ("2031-01", "2031-02")]):
        cl = d.panel.astype(str) + "_" + d.day.astype(str)
        print(f"{nm}: windows {len(d)} | base {d['w0.0'].mean():.4f} | " + " | ".join(
            f"w {wt}: {(d[f'w{wt}'] - d['w0.0']).mean():+.4f} (SE {boot((d[f'w{wt}'] - d['w0.0']).to_numpy(), cl):.4f}, "
            f"z {(d[f'w{wt}'] - d['w0.0']).mean() / max(boot((d[f'w{wt}'] - d['w0.0']).to_numpy(), cl), 1e-12):+.1f})" for wt in WS[1:]))
