"""G158 (11/01, ledger #117b stage 0, follow-up of #117): the inputs are V / vf with vf from the training months, and vf moves per link in the redrawn months. Does a network read that as a
slowdown? Calibration of the saved probabilities on M3 dev by x = the link's March free-flow level / its training level (g154): actual / predicted per bin of x, and b in
logit(y) ~ logit(p) + b (x - 1) (window bootstrap). Onset: the T+30 row of the final blend (G120). Ongoing: all rows of the base networks (c24f_attnglob) and of X1, links
not queued at T only (a jam already there says nothing about the slowdown reading). Hold-out days are not read.
usage: TFB_M3VER=v3 py -B -X utf8 g158_vf_calib.py"""
import os, sys, pickle, importlib.util
os.environ.setdefault("TFB_M3VER", "v3"); sys.dont_write_bytecode = True; R0 = r"C:/Users/user/tfb/"
import numpy as np, pandas as pd
sys.argv = [sys.argv[0], "run"]; spec = importlib.util.spec_from_file_location("g121", R0 + "review/g121_ongoing_oracle.py"); g = importlib.util.module_from_spec(spec); spec.loader.exec_module(g)
G, m3v2 = g.G, g.m3v2
R = pd.read_csv(R0 + "work/t2/g154_vf_month.csv"); tr = R[R.split == "train"].groupby(["panel", "link"]).r.median(); X = {}
for p, d in R[R.month == "2031-03"].groupby("panel"):
    X[p] = d.sort_values("link").r.to_numpy() / tr[p].to_numpy()
BINS = [0, 0.98, 0.99, 1.01, 1.02, 9]; LOG = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))


def table(rows, name):
    D = pd.DataFrame(rows, columns=["w", "x", "p", "y"]); D["b"] = np.digitize(D.x, BINS) - 1; ws = D.w.unique(); ix = {w: i for i, w in enumerate(ws)}; D["i"] = D.w.map(ix)
    P = np.zeros((len(ws), 5)); Y = np.zeros((len(ws), 5)); np.add.at(P, (D.i, D.b), D.p); np.add.at(Y, (D.i, D.b), D.y); Rr = np.random.default_rng(0).multinomial(len(ws), np.full(len(ws), 1 / len(ws)), size=2000)
    rat = (Y.sum(0) / P.sum(0)); bs = (Rr @ Y) / np.maximum(Rr @ P, 1e-9); se = bs.std(axis=0); n = np.bincount(D.b, minlength=5)
    print(f"{name}: cells {len(D)}, windows {len(ws)}")
    for j, lab in enumerate(("x<0.98", "0.98-0.99", "0.99-1.01", "1.01-1.02", ">1.02")):
        print(f"  {lab:10s} cells {n[j]:7d} predicted {P[:, j].sum():9.1f} actual {Y[:, j].sum():8.0f} actual/pred {rat[j]:.3f} (SE {se[j]:.3f})")
    from sklearn.linear_model import LogisticRegression
    Z = np.column_stack([LOG(D.p.to_numpy()), (D.x.to_numpy() - 1) * 100]); f = lambda m_: LogisticRegression(C=1e6, max_iter=300).fit(Z[m_], D.y.to_numpy()[m_]).coef_[0]; c = f(np.ones(len(D), bool)); rng = np.random.default_rng(1); idx = {i: g_.index.to_numpy() for i, g_ in D.groupby("i")}; bb = []
    for _ in range(200):
        bb.append(f(np.concatenate([idx[i] for i in rng.integers(0, len(ws), len(ws))]))[1])
    print(f"  logit(y) ~ a logit(p) + b 100 (x - 1): a {c[0]:.3f}, b {c[1]:+.4f} (SE {np.std(bb):.4f}, z {c[1] / np.std(bb):+.2f})", flush=True)


PN = pickle.load(open(g.W + "g120_pn_dev_onset.pkl", "rb")); on = []
for w, p_ in PN.items():
    m = G.META[w]; y = m["y"].T.astype(bool)[5]; e = m["m"].T.astype(bool)[5]; x = X[m["panel"]]
    for j in np.flatnonzero(e & np.isfinite(x)):
        on.append((w, x[j], p_[5, j], float(y[j])))
table(on, "onset T+30, final blend")
PB = [pickle.load(open(g.W + f"probs_m3v3__c24f_attnglob_s{s}.pkl", "rb")) for s in (0, 1)]; og = {"base": [], "x1": []}
for w in g.DEV:
    m = G.META[w]; i = G.POS[w]; lq = np.asarray(G.X[G.OFF[i]:G.OFF[i + 1], G.CI], float) == 1; y = m["y"].T.astype(bool); e = m["m"].T.astype(bool); x = X[m["panel"]]; far = ~G.dil(lq, 3); b = (PB[0][w] + PB[1][w]) / 2
    kk = [k for k in m3v2.kdist(m["day"]) if (w, k) in g.X1]; x1 = sum(g.X1[(w, k)] for k in kk) / len(kk)
    r_, c_ = np.nonzero(e & far[None, :] & np.isfinite(x)[None, :])
    for a_, b_ in zip(r_, c_):
        og["base"].append((w, x[b_], b[a_, b_], float(y[a_, b_]))); og["x1"].append((w, x[b_], x1[a_, b_], float(y[a_, b_])))
table(og["base"], "ongoing, links at distance > 3 from the jam at T, base network"); table(og["x1"], "same cells, X1")
