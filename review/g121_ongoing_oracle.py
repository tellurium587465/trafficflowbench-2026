"""G121 (10/30, ledger #75, strategy J15): an oracle decomposition of the ongoing IoU loss -> a ceiling for ongoing.
M3 v3 DEV ongoing windows (the hold-out days are not read), final A's ongoing flow = G108 column A (X1 0.8 + LightGBM
0.2 -> cone -> seen x0.85 -> best_set -> far gate; k expectation, simulated blanks). Per (window, k) every error cell
(sel xor y inside the observation mask) carries 1/U (U = cells of the union), so the parts sum to 1 - IoU. Per horizon
row, the connected runs of sel | y (adjacent link index, before the mask):
  (ii)  extent : a run holding a true positive -> all its error cells (distance to the nearest TP 1 / 2 / >=3, side up =
                 lower index (g55: up = -1) / down / in, FP / FN)
  (i)   ending : a run without a TP that meets the jam at T (last_queue runs dilated by 1): one side ended it, the other not
  (iii) place  : a run without a TP away from the jam at T (a missed new jam FN / a spurious one FP)
Ceiling = 1 - (i), movable = (ii) + (iii); LB scale = each loss x (1 - 0.881) / (1 - 0.919).
AUC (same windows): per last_queue run with >= 1 observed cell at T+30, positive = no true cell left in the run at T+30;
scores -u (b13d: flow per lane over the 3 links upstream of the tail, rows T-3..T-1) and -p (final A's probability before
best_set, T+30 row, run mean, k-weighted).
Identity (first): parts sum to 1 - IoU per (window, k) (1e-9) and the window values reproduce G108's column A (1e-12).
usage: TFB_M3VER=v3 py g121_ongoing_oracle.py [time | run]"""
import os, sys, time, pickle
from collections import defaultdict
os.environ.setdefault("TFB_M3VER", "v3")
sys.path.insert(0, r"C:/Users/user/tfb/src"); argv0 = sys.argv[0]; MODE = (sys.argv[1:] or ["run"])[0]
TM = time.perf_counter; CUR = {}; T00 = TM()
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
import t2_final_spec as FS
sys.argv = [argv0, "0,1,2,3"]
import t2_m3_gate as G, t2_m3v2_gate as V, m3v2
from t2_model import best_set


def tick(name, t0):
    CUR[name] = CUR.get(name, 0.0) + TM() - t0


tick("import (M3 meta, X)", T00)
W = G.W; VER = G.VER; LBX = (1 - 0.881) / (1 - 0.919)
t0 = TM()
XA = [pickle.load(open(W + f"probs_m3{VER}v2x1__c24f_attnglob_s{s}.pkl", "rb")) for s in (0, 1)]
X1 = {key: (XA[0][key] + XA[1][key]) / 2 for key in XA[0]}
src = open(r"C:/Users/user/tfb/review/g55_m3_x7_shock.py", encoding="utf-8").read().split("\nrows, st = [], []")[0]
g55 = type(sys)("g55"); exec(compile(src, "g55", "exec"), g55.__dict__)
tick("load X1 probs, g55", t0)
DEV = [w for w in G.WID if G.META[w]["cond"] == "queue_ongoing" and not m3v2.is_holdout(np.array([G.META[w]["day"]]))[0]]


def cone_f(m, k):                                               # as G108 / G93
    _, cone, _ = g55.parts(m["window_id"])
    P, _, q = m3v2._panel(m["panel"]); t = m["t"]; d0 = t - (t % 288) + 60; hid = V._hidden_today(m, k)
    rr = [r for r in range(d0, t) if r not in hid]
    ev_ = q[rr].any(axis=0) if rr else np.zeros(P.n, bool)
    s2, _ = m3v2.seen(m["panel"], t, k, -18, 12)
    return np.where(G.dil(ev_ | s2, 2)[None, :], 1.0, cone)


def runs(v):
    dd = np.diff(np.r_[0, np.asarray(v, int), 0]); return list(zip(np.flatnonzero(dd == 1), np.flatnonzero(dd == -1) - 1))


def og_pred(p_, m, k):                                          # G108 og_pred, mode None; also returns pb (before best_set)
    s, nd = m3v2.seen(m["panel"], m["t"], k, -18, 12)
    pb = p_ * np.where(G.dil(s, 1), 1.0, 0.85)[None, :] if nd else p_.copy()
    sel = best_set(pb.ravel(), np.ones(pb.size, bool)).reshape(pb.shape)
    return V.far_gate(sel, m, k), pb


def decompose(sel, m, q0d):
    """loss parts of one decode (dict key -> loss) and its IoU"""
    y, mk = m["y"].T.astype(bool), m["m"].T.astype(bool); out = defaultdict(float)
    U = ((sel | y) & mk).sum()
    if U == 0:
        return out, 1.0
    for h in range(sel.shape[0]):
        s_, y_ = sel[h], y[h]; tp = s_ & y_
        for a, b in runs(s_ | y_):
            err = np.flatnonzero((s_[a:b + 1] ^ y_[a:b + 1]) & mk[h, a:b + 1]) + a
            if not err.size:
                continue
            tpi = np.flatnonzero(tp[a:b + 1]) + a
            for j in err:
                fp = "fp" if s_[j] else "fn"
                if tpi.size:
                    d = int(np.abs(tpi - j).min()); side = "up" if j < tpi.min() else ("down" if j > tpi.max() else "in")
                    out[f"ii_{fp}_{side}_{min(d, 3)}"] += 1.0 / U
                elif q0d[a:b + 1].any():
                    out[f"i_{fp}"] += 1.0 / U
                else:
                    out[f"iii_{fp}"] += 1.0 / U
    return out, float((sel & y & mk).sum() / U)


def run(limit=None):
    rows, auc, dsum = [], [], 0.0
    for w in DEV[:limit]:
        m = G.META[w]; i = G.POS[w]; t0 = TM(); Xw = np.asarray(G.X[G.OFF[i]:G.OFF[i + 1]])
        lq = np.asarray(Xw[:, G.CI], float) == 1; q0d = G.dil(lq, 1); P, _, _ = m3v2._panel(m["panel"]); tick("window inputs", t0)
        t0 = TM(); pl = FS.lgb_ongoing(Xw); tick("LightGBM", t0)
        r = defaultdict(float, w=w, panel=m["panel"], day=m["day"]); pr5 = {}
        for k, pr in m3v2.kdist(m["day"]).items():
            t0 = TM(); p = (0.8 * X1[(w, k)] + 0.2 * pl) * cone_f(m, k); tick("cone", t0)
            t0 = TM(); sel, pb = og_pred(p, m, k); tick("decode (best_set + far gate)", t0)
            t0 = TM(); parts, iou = decompose(sel, m, q0d); tick("decompose", t0)
            dsum = max(dsum, abs(sum(parts.values()) - (1 - iou)), abs(iou - G.iou(sel, m)))
            r["A"] += pr * iou
            for key, v in parts.items():
                r[key] += pr * v
            pr5[k] = (pr, pb[5])
        t0 = TM(); y, mk = m["y"].T.astype(bool), m["m"].T.astype(bool); ql = P.Q / P.lanes[None, :]; t = m["t"]
        for a, b in runs(lq):
            ob = mk[5, a:b + 1]
            if not ob.any():
                continue
            ups = list(range(max(0, a - 3), a))
            with np.errstate(all="ignore"):
                u = float(np.nanmean(ql[t - 3:t][:, ups])) if ups else np.nan
            pm = sum(pr * v[a:b + 1].mean() for pr, v in pr5.values())
            auc.append(dict(w=w, gone=not (y[5, a:b + 1] & ob).any(), u=u, p=pm, len=b - a + 1))
        tick("AUC rows", t0); rows.append(dict(r))
    return pd.DataFrame(rows).fillna(0.0), pd.DataFrame(auc), dsum


def boot(d, cl, n_boot=2000, seed=0):
    rng = np.random.default_rng(seed); ids = pd.factorize(cl)[0]; M = np.bincount(ids, d); N = np.bincount(ids)
    R = rng.multinomial(len(M), np.full(len(M), 1.0 / len(M)), size=n_boot); return float(((R @ M) / (R @ N)).std())


if __name__ == "__main__":
    D, A, dsum = run(1 if MODE == "time" else None)
    ref = pd.read_csv(W + "g108_rectangles.csv").set_index("w")["A"]
    dmax = float(np.max(np.abs(D.A.to_numpy() - ref.loc[D.w].to_numpy())))
    print(f"identity: parts vs 1 - IoU and IoU vs G.iou max |d| {dsum:.2e} | windows vs G108 column A max |d| {dmax:.2e} "
          f"over {len(D)} windows | mean A {D.A.mean():.4f}")
    print("time per stage (s):", {k: round(v, 2) for k, v in CUR.items()}, f"| total {TM() - T00:.1f}s")
    if MODE == "time":
        per = sum(v for k, v in CUR.items() if not k.startswith(("import", "load")))
        print(f"per window {per:.2f}s -> {len(DEV)} windows {per * len(DEV) / 60:.1f} min + loads"); sys.exit()
    if dsum > 1e-9 or dmax > 1e-12:
        print("IDENTITY FAILED: stop"); sys.exit(1)
    D.to_csv(W + "g121_ongoing_oracle.csv", index=False); A.to_csv(W + "g121_ongoing_auc.csv", index=False)
    cl = D.panel + "_" + D.day.astype(str)
    g = lambda pre: D[[c for c in D.columns if c.startswith(pre)]].sum(axis=1).to_numpy()
    I, II, III = g("i_"), g("ii_"), g("iii_"); E2 = sum(D[c] for c in D.columns if c.startswith("ii_") and c[-1] in "12").to_numpy()
    loss = 1 - D.A.to_numpy()
    print(f"dev ongoing windows {len(D)} | loss 1 - IoU {loss.mean():.4f} (SE {boot(loss, cl):.4f}) = (i) + (ii) + (iii) "
          f"{I.mean() + II.mean() + III.mean():.4f}")
    for nm, v in (("(i) ending", I), ("(ii) extent", II), ("(ii) within 2 cells", E2), ("(iii) place", III), ("(ii)+(iii) movable", II + III)):
        print(f"  {nm:22s} {v.mean():.4f} (SE {boot(v, cl):.4f}) | share of loss {v.mean() / loss.mean():.2f}")
    print("  (i)  FP (truth ended, kept) %.4f | FN (dropped, truth kept) %.4f" % (D.get("i_fp", 0).mean(), D.get("i_fn", 0).mean()))
    print("  (iii) FP (spurious) %.4f | FN (missed new jam) %.4f" % (D.get("iii_fp", 0).mean(), D.get("iii_fn", 0).mean()))
    for fp in ("fp", "fn"):
        for side in ("up", "down", "in"):
            print(f"  (ii) {fp} {side:4s}: " + " | ".join(f"d{d} {D.get(f'ii_{fp}_{side}_{d}', pd.Series(0.0)).mean():.4f}" for d in (1, 2, 3)))
    ceil, mov = 1 - I.mean(), (II + III).mean()
    print(f"ceiling (M3) = 1 - (i) = {ceil:.4f} (SE {boot(I, cl):.4f}) | movable (M3) {mov:.4f} = S {0.15 * mov:+.4f}")
    print(f"LB scale (x{LBX:.2f}): now 0.881 | ceiling {1 - LBX * I.mean():.4f} | movable {LBX * mov:.4f} = S {0.15 * LBX * mov:+.4f}")
    print(f"line: movable >= 0.03 {mov >= 0.03} | (ii) within 2 cells > half of movable {E2.mean() > 0.5 * mov}")
    for nm in ("u", "p"):
        a = A[np.isfinite(A[nm])]
        print(f"AUC 'run gone at T+30' by -{nm}: {roc_auc_score(a.gone, -a[nm]):.3f} (runs {len(a)}, gone {a.gone.mean():.3f})")
