"""G156 (11/01, ledger #116, strategy T2_DEEP_REVIEW 1-4): is M3 dev ongoing a quieter set of windows than the official ones? History-only features (shares of the observed
cells, since the history coverage differs: M3 dev 0.577, official 0.76) on the 356 dev windows and the official ongoing windows of March, April, June (40 each, no labels read).
Logistic regression official vs M3 dev, 5 folds over windows, weights = the M3 weights x p / (1 - p), cut at the 99th percentile. Moves the level of final A's March flow and,
if the line is met, each component: A against A without the far gate, X1 (the same networks with no fine-tuning), cone, LightGBM 0.2, tfm 0.2.
Identity: column A against g132_grid_m3.csv l20_t20_c50_D3 (the same flow, dev days only). g110_tfm5.csv is not read, nor are the hold-out days.
usage: TFB_M3VER=v3 py -B -X utf8 g156_ongoing_reweight.py [dry | run]"""
import os, sys, pickle, importlib.util
os.environ.setdefault("TFB_M3VER", "v3"); sys.dont_write_bytecode = True; RUN = (sys.argv[1:] or ["run"])[0]; R0 = r"C:/Users/user/tfb/"
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score


def module(nm, path):
    spec = importlib.util.spec_from_file_location(nm, path); mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


sys.argv = [sys.argv[0], "run"]; g = module("g121", R0 + "review/g121_ongoing_oracle.py"); g132 = module("g132", R0 + "review/g132_p3_grid.py")
G, V, m3v2, best_set = g.G, g.V, g.m3v2, g.best_set
from cachelib import load, T2_PANELS
from common import REL
FE = ["share", "last", "growth", "persist", "run", "age"]


def feats(V_, E, vcut, t, day0):
    """V_ = the visible layer, rows t-72 .. t-1 of the same day at most"""
    lo = max(t - 72, day0); W_ = V_[lo:t]; O = np.isfinite(W_) & E[lo:t]; Q = O & (W_ <= vcut[None, :]); h = Q[-12:]; oh = O[-12:]
    sh = lambda r: h[r].sum() / max(oh[r].sum(), 1); last = h[-1]; ol = oh[-1]; i = u = 0
    for r in range(-6, -1):
        both = oh[r] & ol; i += (h[r] & last & both).sum(); u += ((h[r] | last) & both).sum()
    best = cur = 0; gap = False
    for j in range(len(last)):
        if last[j]:
            cur += 1; best = max(best, cur); gap = False
        elif ol[j]:
            cur = 0
        elif cur:
            cur += 1 if not gap else 0; gap = True
    age = 0; run_free = 0
    for r in range(len(Q) - 1, -1, -1):
        if Q[r].any():
            run_free = 0
        elif O[r].any():
            run_free += 1
            if run_free == 2:
                break
        age += 1
    return dict(share=sh(slice(None)), last=sh(-1), growth=sh(slice(-3, None)) - sh(slice(-6, -3)), persist=i / max(u, 1), run=best, age=min(age, 60))


def official():
    rows = []
    for sp, tag in (("validation", "March"), ("private", "April"), ("train", "June")):
        for p in T2_PANELS:
            P = load(p, sp); ti = {str(s)[:16]: i for i, s in enumerate(P.times)}; Vis = np.where(P.blank, np.nan, P.V) if sp == "train" else P.V
            for r in pd.read_csv(REL / "task2" / p / sp / "window_index.csv").itertuples():
                if r.condition == "queue_ongoing":
                    t = ti[r.forecast_origin[:16]]; L = Vis if sp != "train" else np.vstack([Vis[:t - 12], P.V[t - 12:t]])
                    rows.append(dict(src=tag, panel=p, **feats(L, P.E, P.vcut, t, t - t % 288)))
            load.cache_clear()
    return pd.DataFrame(rows)


def m3_rows(dev):
    rows = []
    for w in dev:
        m = G.META[w]; P, _, _ = m3v2._panel(m["panel"]); t = m["t"]; rows.append(dict(w=w, src="M3", panel=m["panel"], **feats(P.V, P.E, P.vcut, t, t - t % 288)))
    return pd.DataFrame(rows)


def odds(M, O, seed=0):
    X = np.vstack([M[FE].to_numpy(float), O[FE].to_numpy(float)]); X[:, 4:] = np.log1p(X[:, 4:]); y = np.r_[np.zeros(len(M)), np.ones(len(O))]; p = np.zeros(len(y))
    for a, b in StratifiedKFold(5, shuffle=True, random_state=seed).split(X, y):
        p[b] = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=1000)).fit(X[a], y[a]).predict_proba(X[b])[:, 1]
    o = p[: len(M)] / (1 - p[: len(M)]); return np.minimum(o, np.quantile(o, 0.99)), roc_auc_score(y, p)


def score(dev):
    PL = [pickle.load(open(g.W + f"probs_m3v3__tfm4_s{s}.pkl", "rb")) for s in range(5)]; PB = [pickle.load(open(g.W + f"probs_m3v3__c24f_attnglob_s{s}.pkl", "rb")) for s in (0, 1)]
    CC = ("A", "no_gate", "no_x1", "no_cone", "no_lgb", "no_tfm"); cols = {c: {} for c in CC}
    for w in dev:
        m = G.META[w]; i = G.POS[w]; Xw = np.asarray(G.X[G.OFF[i]:G.OFF[i + 1]]); pl = g.FS.lgb_ongoing(Xw); pt = sum(d[w] for d in PL) / 5; x0 = (PB[0][w] + PB[1][w]) / 2
        lq = np.asarray(Xw[:, G.CI], float) == 1; P, _, q = m3v2._panel(m["panel"]); t = m["t"]; d0 = t - (t % 288) + 60; _, cone, _ = g.g55.parts(w); nr = g132.nears(lq, P.n)[3]; sc = dict.fromkeys(CC, 0.0)
        for k, pr in m3v2.kdist(m["day"]).items():
            hid = V._hidden_today(m, k); rr = [r for r in range(d0, t) if r not in hid]; ev_ = q[rr].any(axis=0) if rr else np.zeros(P.n, bool); s2, _ = m3v2.seen(m["panel"], t, k, -18, 12)
            ex = G.dil(ev_ | s2, 2); s, nd = m3v2.seen(m["panel"], t, k, -18, 12); sw = np.where(G.dil(s, 1), 1.0, 0.85)[None, :] if nd else 1.0; x1 = g.X1[(w, k)]; cf = np.where(ex[None, :], 1.0, np.where(cone < 0.999, 0.5, 1.0))

            def dec(x, wt, wl, c):
                pb = ((1 - wl) * ((1 - wt) * x + wt * pt) + wl * pl) * c * sw; return best_set(pb.ravel(), np.ones(pb.size, bool)).reshape(pb.shape)


            base = dec(x1, 0.2, 0.2, cf); sets = dict(A=g132.gate(base, ex, nr), no_gate=base, no_x1=g132.gate(dec(x0, 0.2, 0.2, cf), ex, nr), no_cone=g132.gate(dec(x1, 0.2, 0.2, 1.0), ex, nr),
                                                       no_lgb=g132.gate(dec(x1, 0.2, 0.0, cf), ex, nr), no_tfm=g132.gate(dec(x1, 0.0, 0.2, cf), ex, nr))
            for c in CC:
                sc[c] += pr * G.iou(sets[c], m)
        for c in CC:
            cols[c][w] = sc[c]
    return V.frame(cols).dropna(subset=["A"]).reset_index(drop=True)


def wmean(a, w):
    return float((a * w).sum() / w.sum())


def boot_move(a, w0, w1, cl, n_boot=2000, seed=0):
    ids = pd.factorize(cl)[0]; n = ids.max() + 1; f = lambda w: (np.bincount(ids, a * w, n), np.bincount(ids, w, n)); (A0, B0), (A1, B1) = f(w0), f(w1)
    R = np.random.default_rng(seed).multinomial(n, np.full(n, 1 / n), size=n_boot); l0 = (R @ A0) / (R @ B0); l1 = (R @ A1) / (R @ B1); return float(l0.std()), float(l1.std()), float((l1 - l0).std())


if __name__ == "__main__":
    dev = g.DEV[:5] if RUN == "dry" else g.DEV; D = score(dev); ref = pd.read_csv(g.W + "p3_grid/g132_grid_m3.csv").set_index("w")["l20_t20_c50_D3"]
    da = float(np.abs(ref.loc[D.w].to_numpy() - D.A.to_numpy()).max()); print(f"{len(D)} windows | identity A against g132 base: max |d| {da:.2e}", flush=True)
    if da > 1e-9:
        print("IDENTITY FAILED: stop"); sys.exit(1)
    O = official(); M = m3_rows(list(D.w)); print("official windows", O.groupby("src").size().to_dict(), flush=True)
    print(pd.concat([O.groupby("src")[FE].mean(), M.groupby("src")[FE].mean()]).round(3).to_string(), flush=True)
    if RUN == "dry":
        sys.exit()
    cl = (D.panel + "_" + D.day.astype(str)).to_numpy(); w0 = {t: m3v2.weights(D, t)[0] for t in ("validation", "private")}; A = D.A.to_numpy()
    print(f"level of A on M3 dev ongoing: unweighted {A.mean():.4f}, March composition {wmean(A, w0['validation']):.4f}, April composition {wmean(A, w0['private']):.4f}", flush=True)
    W1 = {}
    for nm, sel, base in (("all official", ("March", "April", "June"), "validation"), ("March only", ("March",), "validation"), ("April only", ("April",), "private")):
        o, auc = odds(M, O[O.src.isin(sel)]); w1 = w0[base] * o; w1 /= w1.sum(); W1[nm] = (w0[base], w1); s0, s1, sm = boot_move(A, w0[base], w1, cl)
        print(f"{nm}: AUC {auc:.3f} | ESS {1 / np.sum(w0[base] ** 2):.0f} -> {1 / np.sum(w1 ** 2):.0f} | level {wmean(A, w0[base]):.4f} (SE {s0:.4f}) -> {wmean(A, w1):.4f} (SE {s1:.4f}) | move {wmean(A, w1) - wmean(A, w0[base]):+.4f} (SE {sm:.4f}, z {(wmean(A, w1) - wmean(A, w0[base])) / max(sm, 1e-9):+.2f})", flush=True)
    D.assign(odds_all=W1["all official"][1] / W1["all official"][0]).to_csv(g.W + "g156_m3_frame.csv", index=False); pd.concat([O, M]).to_csv(g.W + "g156_feats.csv", index=False)
    for c in ("no_gate", "no_x1", "no_cone", "no_lgb", "no_tfm"):
        d = A - D[c].to_numpy(); out = [f"{c:8s} A minus it: raw {d.mean():+.4f}"]
        for nm, (a_, b_) in W1.items():
            s0, s1, sm = boot_move(d, a_, b_, cl); out.append(f"{nm} {wmean(d, a_):+.4f} ({s0:.4f}) -> {wmean(d, b_):+.4f} ({s1:.4f})")
        print(" | ".join(out), flush=True)
