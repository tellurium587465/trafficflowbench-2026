"""G102 (10/24, X lane 1a, LITERATURE_SCAN_20261024 §1 companion): is the final A's onset at its own ceiling?
On the M3 v3 dev onset windows (hold-out excluded), the final A's onset probabilities (q24f 8 seeds CFG5 + physics onset with
the month-adaptive a, as G93's "phys" column) are decoded as in production (evidence rule "cur" -> best_set) and scored at
T+30 as the instrument does. Per window and k (the k expectation of m3v2.kdist):
  realized = IoU(decoded set, truth)
  expected = E[IoU(decoded set, Y)], Y_j ~ independent Bernoulli(p_j) on the eligible T+30 cells (Monte Carlo), for the raw
             blend p and for the evidence-weighted pb the decoder sees
Reading (pre-declared): realized ~ expected -> the probabilities are calibrated at the window level and their sharpness is
the limit (the model is at its own ceiling; more room needs new information); realized << expected -> overconfident on the
wrong site (room for location work). Diagnostic only, no adoption decision.
usage: TFB_M3VER=v3 python g102_onset_selfcal.py"""
import os, sys, pickle
os.environ.setdefault("TFB_M3VER", "v3")
sys.path.insert(0, r"C:/Users/user/tfb/src"); argv0 = sys.argv[0]
import numpy as np, pandas as pd, lightgbm as lgb
import t2_phys_onset as PO
sys.argv = [argv0, "0,1,2,3"]
import t2_m3_gate as G, t2_m3v2_gate as V, m3v2
from t2_model import best_set
from common import NN

W = G.W; VER = G.VER; PRE = G.PRE
ld = lambda f: pickle.load(open(W + f, "rb"))
CF5 = (("attnglob", 2.0), ("attnwp05", 2.0), ("attn1", 2.0), ("drop1", 1.0), ("r13", 1.0))
ON = [(f"q24f_{c}_s{s}", w) for s in range(8) for c, w in CF5]; PRO = {t: ld(f"probs_{PRE}__{t}.pkl") for t, _ in ON}
X = np.load(f"{NN}/nn_test_m3{VER}/X.npy", mmap_mode="r"); OFF = np.load(f"{NN}/nn_test_m3{VER}/off.npy")
phys = lgb.Booster(model_file=W + "physon_base.txt"); ev = PO.month_events("validation", PO.rarity_table())
rng = np.random.default_rng(102); NS = 4000


def eiou(sel, p, mk):
    """E[IoU(sel, Y)] with Y_j ~ Bernoulli(p_j) independently on the eligible cells mk (T+30 row)"""
    s, pp = sel[mk], np.clip(p[mk], 0, 1)
    Y = rng.random((NS, len(pp))) < pp[None, :]
    u = (Y | s[None, :]).sum(1); i = (Y & s[None, :]).sum(1)
    return float(np.where(u == 0, 1.0, i / np.maximum(u, 1)).mean())


rows, CELL = [], []
for i, w in enumerate(G.WID):
    m = G.META[w]
    if m["cond"] != "queue_onset" or m3v2.is_holdout(np.array([m["day"]]))[0]:
        continue
    Xw = np.asarray(X[OFF[i]:OFF[i + 1]])
    p = sum(wt * PRO[t][w] for t, wt in ON) / sum(wt for _, wt in ON); a = PO.adaptive_a(ev, m["day"])
    p = p.copy(); p[5] = (1 - a) * p[5] + a * phys.predict(PO.feats(Xw))
    mk = m["m"].T.astype(bool)[-1]; y = m["y"].T.astype(bool)[-1]
    r = dict(w=w, panel=m["panel"], day=m["day"], hour=int(m["t"] % 288) // 12, ntrue=int((y & mk).sum()), real=0.0, exp_p=0.0, exp_pb=0.0,
             mass_true=float(p[5][y & mk].sum() / max(p[5][mk].sum(), 1e-9)), pmax=float(p[5][mk].max()) if mk.any() else 0.0)
    CELL.append(np.column_stack([p[5][mk], y[mk]]))
    for k, pr in m3v2.kdist(m["day"]).items():
        pb = p.copy()                                            # V.onset_pred's evidence rule "cur", kept explicit here
        s, nd = m3v2.seen(m["panel"], m["t"], k, -6, 12)
        if nd:
            pb[-1] = pb[-1] * np.where(G.dil(s, 2), 1.0, 0.25)
        sel = best_set(pb.ravel(), np.ones(pb.size, bool)).reshape(pb.shape)
        assert (sel == V.onset_pred(p, m, "cur", k)).all()
        r["real"] += pr * G.iou(sel, m); r["exp_p"] += pr * eiou(sel[-1], p[-1], mk); r["exp_pb"] += pr * eiou(sel[-1], pb[-1], mk)
        r["c_sel"] = r.get("c_sel", 0.0) + pr * float(p[5][sel[-1] & mk].max()) if (sel[-1] & mk).any() else r.get("c_sel", 0.0)
    rows.append(r)
D = pd.DataFrame(rows); D["fam"] = D.panel.str.rsplit("_", n=1).str[0]
D.to_csv(W + "g102_onset_selfcal.csv", index=False)
bs = lambda v: np.std([rng.choice(v, len(v)).mean() for _ in range(2000)])
d = (D.real - D.exp_p).to_numpy()
print(f"M3 {VER} dev onset windows {len(D)}: realized {D.real.mean():.4f} | expected (raw p) {D.exp_p.mean():.4f} | expected (pb) {D.exp_pb.mean():.4f} | "
      f"realized - expected(p) {d.mean():+.4f} (SE {bs(d):.4f})")
print(D.groupby("fam")[["real", "exp_p", "exp_pb"]].mean().round(3).assign(n=D.groupby("fam").size()).to_string())
D["bin"] = pd.cut(D.exp_p, [0, 0.5, 0.8, 0.95, 1.0001], right=False)
print(D.groupby("bin", observed=True)[["real", "exp_p"]].agg(["mean", "count"]).round(3).to_string())
C = np.vstack(CELL); cb = pd.cut(C[:, 0], [0, .05, .2, .4, .6, .8, .95, 1.0001], right=False)
print("cell reliability at T+30 (p bin: mean p, observed rate, cells):")
print(pd.DataFrame({"p": C[:, 0], "y": C[:, 1], "b": cb}).groupby("b", observed=True).agg(p=("p", "mean"), obs=("y", "mean"), n=("y", "size")).round(3).to_string())
D["hit"] = (D.real >= 0.2).astype(float); D["cbin"] = pd.cut(D.c_sel, [0, .5, .8, .95, 1.0001], right=False)
print("window hit (realized >= 0.2) vs the decoded set's top p:")
print(D.groupby("cbin", observed=True).agg(top_p=("c_sel", "mean"), hit=("hit", "mean"), n=("hit", "size")).round(3).to_string())
miss = D[D.real < 0.2]
print(f"missed (realized < 0.2): {len(miss)} windows; their expected(p) mean {miss.exp_p.mean():.3f}, pmax mean {miss.pmax.mean():.3f}, "
      f"share of p-mass on the true cells {miss.mass_true.mean():.3f} (hits: {D[D.real >= 0.2].mass_true.mean():.3f})")
