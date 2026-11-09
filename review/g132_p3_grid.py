"""G132 (10/31, ledger #87, strategy P3): a joint grid over the hand-set constants of final A's ongoing flow.
p0 = (1 - wl) ((1 - wt) X1 + wt p_tfm) + wl LightGBM -> cone factor c -> seen x0.85 -> best_set -> far gate D.
wl, wt in (0.1, 0.2, 0.3), c in (0.3, 0.5, 0.7), D in (2, 3, 4): 81 cells, the base cell = final A6's ongoing (0.2, 0.2, 0.5, 3).
  m3    : M3 v3 DEV ongoing 356 windows (held-out days not read), chosen on odd days, read on even days
  train : 2031-01 / 2031-02 selector-like ongoing windows, G119 SCG path: only c and D exist there (9 cells)
usage: TFB_M3VER=v3 py g132_p3_grid.py [m3 | train]   (G132_DRY=1: 5 windows)"""
import os, sys, time, pickle, importlib.util, itertools
from collections import defaultdict
MODE = (sys.argv[1:] or ["m3"])[0]; DRY = os.environ.get("G132_DRY", "") == "1"
os.environ.setdefault("TFB_M3VER", "v3"); R0 = r"C:/Users/user/tfb/"
import numpy as np, pandas as pd
TM = time.perf_counter; T00 = TM(); OUT = R0 + "work/t2/p3_grid/"; os.makedirs(OUT, exist_ok=True)
WL = WT = (0.1, 0.2, 0.3); CS = (0.3, 0.5, 0.7); DS = (2, 3, 4); BASE = "l20_t20_c50_D3"
name = lambda l, t, c, d: f"l{int(round(l * 100))}_t{int(round(t * 100))}_c{int(c * 100)}_D{d}"


def module(nm, path):
    spec = importlib.util.spec_from_file_location(nm, path); mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


def boot(d, cl, n_boot=2000, seed=0):
    rng = np.random.default_rng(seed); ids = pd.factorize(cl)[0]; M = np.bincount(ids, d); N = np.bincount(ids)
    Rm = rng.multinomial(len(M), np.full(len(M), 1.0 / len(M)), size=n_boot); return float(((Rm @ M) / (Rm @ N)).std())


def contig_near(sel, near):
    """t2_m3v2_gate.far_gate: selected stretches touching the near zone stay"""
    C = np.zeros_like(sel)
    for r_ in range(sel.shape[0]):
        j = 0; row = sel[r_]; n = row.size
        while j < n:
            if row[j]:
                e = j
                while e + 1 < n and row[e + 1]:
                    e += 1
                if near[j:e + 1].any():
                    C[r_, j:e + 1] = True
                j = e + 1
            else:
                j += 1
    return C


def gate(sel, ev, near):
    return sel & ~(sel & ~near[None, :] & ~ev[None, :] & ~contig_near(sel, near))


def nears(lq, n):
    out = {}
    for D in DS:
        nr = np.zeros(n, bool)
        for j in np.flatnonzero(lq):
            nr[max(0, j - D):j + D + 1] = True
        out[D] = nr
    return out


def m3():
    g = module("g121", R0 + "review/g121_ongoing_oracle.py"); G, V, m3v2, best_set = g.G, g.V, g.m3v2, g.best_set
    PL = [pickle.load(open(g.W + f"probs_m3v3__tfm4_s{s}.pkl", "rb")) for s in range(5)]; DEV = g.DEV[:5] if DRY else g.DEV
    NM = [name(l, t, c, d) for l, t, c, d in itertools.product(WL, WT, CS, DS)]; cols = {c: {} for c in NM}
    for w in DEV:
        m = G.META[w]; i = G.POS[w]; Xw = np.asarray(G.X[G.OFF[i]:G.OFF[i + 1]]); pl = g.FS.lgb_ongoing(Xw); pt = sum(d[w] for d in PL) / 5; lq = np.asarray(Xw[:, G.CI], float) == 1
        P, _, q = m3v2._panel(m["panel"]); t = m["t"]; d0 = t - (t % 288) + 60; _, cone, _ = g.g55.parts(w); nr = nears(lq, P.n); sc = defaultdict(float)
        for k, pr in m3v2.kdist(m["day"]).items():
            hid = V._hidden_today(m, k); rr = [r for r in range(d0, t) if r not in hid]; ev_ = q[rr].any(axis=0) if rr else np.zeros(P.n, bool); s2, _ = m3v2.seen(m["panel"], t, k, -18, 12)
            ex = G.dil(ev_ | s2, 2); s, nd = m3v2.seen(m["panel"], t, k, -18, 12); sw = np.where(G.dil(s, 1), 1.0, 0.85)[None, :] if nd else 1.0; x1 = g.X1[(w, k)]
            for l, t_ in itertools.product(WL, WT):
                p0 = (1 - l) * ((1 - t_) * x1 + t_ * pt) + l * pl
                for c in CS:
                    pb = p0 * np.where(ex[None, :], 1.0, np.where(cone < 0.999, c, 1.0)) * sw; sel = best_set(pb.ravel(), np.ones(pb.size, bool)).reshape(pb.shape)
                    for D in DS:
                        sc[name(l, t_, c, D)] += pr * G.iou(gate(sel, ex, nr[D]), m)
        for c in sc:
            cols[c][w] = sc[c]
    ref = pd.read_csv(g.W + "g110_tfm5.csv").set_index("w")["T"]; da = max(abs(cols[BASE][w] - ref[w]) for w in DEV); print(f"identity base cell vs g110_tfm5 T max |d| {da:.2e} | {TM() - T00:.0f}s", flush=True)
    if da > 1e-9:
        print("IDENTITY FAILED: stop"); sys.exit(1)
    if DRY:
        return
    D_ = V.frame(cols).dropna(subset=[BASE]); D_.to_csv(OUT + "g132_grid_m3.csv", index=False); cl = D_.panel + "_" + D_.day.astype(str); odd = (D_.day % 2 == 1).to_numpy()
    r = pd.DataFrame([dict(cell=c, dev=(D_[c] - D_[BASE]).mean(), odd=(D_[c] - D_[BASE])[odd].mean(), even=(D_[c] - D_[BASE])[~odd].mean()) for c in NM]).sort_values("odd", ascending=False); r.to_csv(OUT + "g132_grid_summary.csv", index=False)
    print(r.head(6).round(4).to_string(index=False)); best = r.cell.iloc[0]; x = (D_[best] - D_[BASE]).to_numpy(); se = boot(x[~odd], cl[~odd]); sa = boot(x, cl)
    print(f"selected on odd days: {best}; even days {x[~odd].mean():+.4f} (SE {se:.4f}, z {x[~odd].mean() / max(se, 1e-12):+.2f}) | all dev {x.mean():+.4f} (SE {sa:.4f}, z {x.mean() / max(sa, 1e-12):+.2f}, k = 81 E[max] 2.46) | base dev {D_[BASE].mean():.4f}")
    for ax, vals, tag in (("l", WL, "wl"), ("t", WT, "wt"), ("c", CS, "c"), ("D", DS, "D")):
        print(tag, {v: round(float(r[r.cell.str.contains(f"{ax}{int(round(v * 100)) if ax != 'D' else v}(?:_|$)")].dev.mean()), 4) for v in vals})


def train():
    g = module("g119", R0 + "review/g119_month_postproc.py"); T, G, m3v2 = g.T, g.G, g.m3v2
    OOF = [pickle.load(open(g.W + f"oof_q25_nomem_s{s}.pkl", "rb")) for s in (3, 4)]; rows = []
    for mo in g.MONTHS:
        ks = np.flatnonzero(T.META["is_sel"] & (g.MON == mo) & np.array([str(c) == "queue_ongoing" for c in T.META["cond"]]))
        for p in sorted({str(T.META["panel"][k]) for k in ks}):
            P = g.load(p, "train"); key = (mo, p, "mask"); g.CTX.clear(); g.sim_blank.cache_clear()
            g.CTX[key] = g.Ctx(key, P, np.where(P.blank, np.nan, P.V), P.V); c_ = g.CTX[key]; kp = [k for k in ks if str(T.META["panel"][k]) == p]
            for k in kp[:2] if DRY else kp:
                t = int(T.META["t"][k]); lab, el = T.LAB[k]; y, e = lab.astype(bool), el.astype(bool); Xw = np.asarray(g.XN[g.OFFN[k]:g.OFFN[k + 1]]); lq = np.asarray(Xw[:, G.CI], float) == 1
                p_ = (OOF[0][k] + OOF[1][k]) / 2; dom = int(P.times[t][8:10]) - 1; cone = g.cone_parts(c_, t, lq, P.length); nr = nears(lq, P.n); r = defaultdict(float, w=int(k), month=mo, panel=p, day=dom)
                for kk, pr in m3v2.kdist(dom, cap=3).items():
                    ev_ = g.today_ev(c_, t, kk); s2, _ = g.seen(c_, t, kk, -18, 12); ex = G.dil(ev_ | s2, 2); s, nd = g.seen(c_, t, kk, -18, 12); sw = np.where(G.dil(s, 1), 1.0, 0.85)[None, :] if nd else 1.0
                    for c in CS:
                        pb = p_ * np.where(ex[None, :], 1.0, np.where(cone < 0.999, c, 1.0)) * sw; sel = g.decode(pb)
                        for D in DS:
                            r[f"c{int(c * 100)}_D{D}"] += pr * g.iou(gate(sel, ex, nr[D]), y, e, "queue_ongoing")
                rows.append(dict(r))
            g.load.cache_clear()
        print(mo, "done", len(rows), f"{TM() - T00:.0f}s", flush=True)
        if DRY:
            break
    Df = pd.DataFrame(rows); ref = pd.read_csv(g.W + "g119_month_postproc.csv", dtype={"w": str}); ref = ref[(ref.vis == "mask") & (ref.cond == "queue_ongoing")].set_index(["month", "w"]).SCG
    da = float(np.max(np.abs(Df["c50_D3"].to_numpy() - np.array([ref.loc[(r.month, str(r.w))] for r in Df.itertuples()])))); print(f"identity c50_D3 vs G119 SCG max |d| {da:.2e} | {TM() - T00:.0f}s", flush=True)
    if da > 1e-9:
        print("IDENTITY FAILED: stop"); sys.exit(1)
    if DRY:
        return
    Df.to_csv(OUT + "g132_grid_train.csv", index=False)
    for mo in g.MONTHS:
        d = Df[Df.month == mo]; cl = d.panel + "_" + d.day.astype(str); print(mo, len(d), "base", round(d["c50_D3"].mean(), 4), " | ".join(f"{c} {(d[c] - d['c50_D3']).mean():+.4f} (z {(d[c] - d['c50_D3']).mean() / max(boot((d[c] - d['c50_D3']).to_numpy(), cl), 1e-12):+.1f})" for c in Df.columns if c.startswith("c") and c != "c50_D3"))


if __name__ == "__main__":
    {"m3": m3, "train": train}[MODE]()
