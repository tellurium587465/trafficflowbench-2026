"""G122 (10/30, ledger #76, strategy J16): trim the upstream end of decoded ongoing runs at known sites.
Rule: per horizon row, each run [a, b] of the decoded set (head = b, upstream = lower index) whose head is a known site
(b, b-1, b+1 in that order, as G108's snap) loses the cells a .. tail-1 upstream of the site's modal tail whose decode
probability pb (the one given to best_set) is below tau; tau in {0.5, 0.6, 0.7, 0.8} (and 0 = the identity).
Sites = b13_events (training months, birth row observed) with the b13c criteria (n >= 30, modal-tail share >= 0.6): all
months for March (= work/b13_sites.parquet), the month left out for January / February.
  m3    : M3 v3 DEV ongoing windows, final A's flow (the G121 path), base = G108 column A
  train : 2031-01 / 2031-02 selector-like ongoing windows, the G119 path (month-CV OOF q25_nomem s3/s4, visibility mask):
          raw (best_set only, no March constant; B's April type) and SCG (A type, record only); base = g119_month_postproc.csv
Estimator: paired mean, SE = cluster (panel x day) bootstrap.
usage: py g122_tail_trim.py [time_m3 | m3 | time_train | train]"""
import os, sys, time, pickle, importlib.util
from collections import defaultdict
MODE = (sys.argv[1:] or ["m3"])[0]; TAUS = (0.0, 0.5, 0.6, 0.7, 0.8)
import numpy as np, pandas as pd
TM = time.perf_counter; CUR = {}; T00 = TM()
R = r"C:/Users/user/tfb/"


def tick(name, t0):
    CUR[name] = CUR.get(name, 0.0) + TM() - t0


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path); mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


def sites(leave_out=None):
    E = pd.read_parquet(R + "work/b13_events.parquet"); E = E[E.raw0]
    if leave_out:
        E = E[~E.day.str.startswith(leave_out)]
    S = E.groupby(["panel", "head"]).tail0.agg(n="size", tail=lambda s: s.mode().iloc[0], share=lambda s: (s == s.mode().iloc[0]).mean()).reset_index()
    S = S[(S.n >= 30) & (S.share >= 0.6)]; return {(r.panel, int(r.head)): int(r.tail) for r in S.itertuples()}


def runs(v):
    dd = np.diff(np.r_[0, np.asarray(v, int), 0]); return list(zip(np.flatnonzero(dd == 1), np.flatnonzero(dd == -1) - 1))


def trim(sel, pb, panel, site, tau):
    t0 = TM(); out = sel.copy()
    for h in range(sel.shape[0]):
        for a, b in runs(sel[h]):
            hd = next((x for x in (b, b - 1, b + 1) if (panel, x) in site), None)
            if hd is not None and a < site[(panel, hd)]:
                js = np.arange(a, site[(panel, hd)]); out[h, js[pb[h, js] < tau]] = False
    tick("trim", t0); return out


def boot(d, cl, n_boot=2000, seed=0):
    rng = np.random.default_rng(seed); ids = pd.factorize(cl)[0]; M = np.bincount(ids, d); N = np.bincount(ids)
    Rm = rng.multinomial(len(M), np.full(len(M), 1.0 / len(M)), size=n_boot); return float(((Rm @ M) / (Rm @ N)).std())


def m3(limit=None):
    g = module("g121", R + "review/g121_ongoing_oracle.py"); tick("import (g121: M3 meta, X1, g55)", T00)
    site = sites(); S0 = pd.read_parquet(R + "work/b13_sites.parquet")
    assert site == {(r.panel, int(r.head)): int(r.tail) for r in S0.itertuples()}, "site table differs from b13_sites"
    rows = []
    for w in g.DEV[:limit]:
        m = g.G.META[w]; i = g.G.POS[w]; t0 = TM(); Xw = np.asarray(g.G.X[g.G.OFF[i]:g.G.OFF[i + 1]]); pl = g.FS.lgb_ongoing(Xw); tick("inputs + LightGBM", t0)
        r = defaultdict(float, w=w, panel=m["panel"], day=m["day"])
        for k, pr in g.m3v2.kdist(m["day"]).items():
            t0 = TM(); p = (0.8 * g.X1[(w, k)] + 0.2 * pl) * g.cone_f(m, k); sel, pb = g.og_pred(p, m, k); tick("cone + decode", t0)
            t0 = TM(); r["A"] += pr * g.G.iou(sel, m); tick("score", t0)
            for tau in TAUS:
                s2 = trim(sel, pb, m["panel"], site, tau); t0 = TM()
                r[f"t{tau}"] += pr * g.G.iou(s2, m); r[f"n{tau}"] += pr * int((s2 != sel).sum()); tick("score", t0)
        rows.append(dict(r))
    D = pd.DataFrame(rows); ref = pd.read_csv(g.W + "g108_rectangles.csv").set_index("w")["A"]
    return D, float(np.max(np.abs(D.A.to_numpy() - ref.loc[D.w].to_numpy()))), "panel"


def train(limit=None):
    g = module("g119", R + "review/g119_month_postproc.py"); tick("import (g119: train evaluator, X)", T00)
    T, G, m3v2 = g.T, g.G, g.m3v2
    OOF = [pickle.load(open(g.W + f"oof_q25_nomem_s{s}.pkl", "rb")) for s in (3, 4)]
    REF = pd.read_csv(g.W + "g119_month_postproc.csv", dtype={"w": str}); REF = REF[(REF.vis == "mask") & (REF.cond == "queue_ongoing")].set_index(["month", "w"])
    rows = []; dref = 0.0
    for mo in g.MONTHS:
        site = sites(mo); ks = [k for k in np.flatnonzero(T.META["is_sel"] & (g.MON == mo)) if str(T.META["cond"][k]) == "queue_ongoing"]
        for p in sorted({str(T.META["panel"][k]) for k in ks}):
            t0 = TM(); P = g.load(p, "train"); key = (mo, p, "mask"); g.CTX.clear(); g.sim_blank.cache_clear()
            g.CTX[key] = g.Ctx(key, P, np.where(P.blank, np.nan, P.V), P.V); c = g.CTX[key]; tick("load + ctx", t0)
            for k in [k for k in ks if str(T.META["panel"][k]) == p]:
                t = int(T.META["t"][k]); dom = int(P.times[t][8:10]) - 1; lab, el = T.LAB[k]; y, e = lab.astype(bool), el.astype(bool)
                Xw = np.asarray(g.XN[g.OFFN[k]:g.OFFN[k + 1]]); lq = np.asarray(Xw[:, G.CI], float) == 1; p_ = (OOF[0][k] + OOF[1][k]) / 2
                r = defaultdict(float, w=int(k), month=mo, panel=p, day=dom); cone = g.cone_parts(c, t, lq, P.length)
                sel = g.decode(p_); r["raw"] = g.iou(sel, y, e, "queue_ongoing")
                for tau in TAUS:
                    s2 = trim(sel, p_, p, site, tau); r[f"raw_t{tau}"] = g.iou(s2, y, e, "queue_ongoing"); r[f"raw_n{tau}"] = int((s2 != sel).sum())
                for kk, pr in m3v2.kdist(dom, cap=3).items():
                    pc = p_ * g.cone_f(c, cone, t, kk); s_, nd = g.seen(c, t, kk, -18, 12)
                    pb = pc * np.where(G.dil(s_, 1), 1.0, 0.85)[None, :] if nd else pc.copy()
                    sel = g.far_gate(c, g.decode(pb), t, kk, lq); r["SCG"] += pr * g.iou(sel, y, e, "queue_ongoing")
                    for tau in TAUS:
                        s2 = trim(sel, pb, p, site, tau); r[f"SCG_t{tau}"] += pr * g.iou(s2, y, e, "queue_ongoing"); r[f"SCG_n{tau}"] += pr * int((s2 != sel).sum())
                ref = REF.loc[(mo, str(k))]; dref = max(dref, abs(r["raw"] - ref.raw), abs(r["SCG"] - ref.SCG)); rows.append(dict(r))
                if limit and len(rows) >= limit:
                    return pd.DataFrame(rows), dref, "panel"
            g.load.cache_clear()
    return pd.DataFrame(rows), dref, "panel"


if __name__ == "__main__":
    lim = 2 if MODE.startswith("time") else None
    D, dref, _ = m3(lim) if MODE.endswith("m3") else train(lim)
    base_cols = [("A", "t")] if MODE.endswith("m3") else [("raw", "raw_t"), ("SCG", "SCG_t")]
    d0 = max(float(np.max(np.abs(D[f"{p}0.0"] - D[b]))) for b, p in base_cols)
    print(f"identity: base vs reference max |d| {dref:.2e} | tau = 0 vs base max |d| {d0:.2e} | windows {len(D)}")
    print("time per stage (s):", {k: round(v, 2) for k, v in CUR.items()}, f"| total {TM() - T00:.1f}s")
    if MODE.startswith("time"):
        sys.exit()
    if dref > 1e-12 or d0 > 0:
        print("IDENTITY FAILED: stop"); sys.exit(1)
    D.to_csv(R + f"work/t2/g122_tail_trim_{MODE}.csv", index=False)
    groups = [("2031-03 dev", D)] if MODE == "m3" else [(mo, D[D.month == mo]) for mo in ("2031-01", "2031-02")]
    for nm, d in groups:
        cl = d.panel.astype(str) + "_" + d.day.astype(str)
        for b, p in base_cols:
            nc = "n" if b == "A" else f"{b}_n"
            print(f"{nm} {b}: windows {len(d)} | base {d[b].mean():.4f}")
            for tau in TAUS[1:]:
                x = (d[f"{p}{tau}"] - d[b]).to_numpy(); se = boot(x, cl)
                print(f"  tau {tau}: gain {x.mean():+.5f} (SE {se:.5f}, z {x.mean() / se if se > 0 else 0:+.2f}) | windows changed "
                      f"{int((d[f'{nc}{tau}'] > 0).sum())} | cells removed per window {d[f'{nc}{tau}'].mean():.2f} | wins {int((x > 1e-12).sum())} losses {int((x < -1e-12).sum())}")
