"""M3 validation, second batch (METAGAME_REPLY §12): V7 weight robustness, V8 label noise, V11 fidelity of the simulated
blanks, and the panel heterogeneity of each adopted part (DerSimonian-Laird random effects over panels, I^2; I^2 > 50% =
"one-panel phenomenon" flag -> kept out of the final B). Uses the v2 score files already written.
  V7  : sign of each adopted part under raking (private), propensity-score weights (logistic model: official April window
        vs M3 window on hour, weekend, base max probability, set size, visible queue size; weights = p / (1 - p)),
        and no weights
  V8  : share of labelled cells within +-2% of the 0.6 v_f threshold; each part re-scored with those cells' labels flipped
        is not done here (needs re-scoring); reported: the share, and the parts' differences restricted to windows
        whose label set has no near-threshold cell
  V11 : the simulated blanks on the evidence slots of M3 windows vs the official March windows' real blanks on their own
        evidence slots (share of evidence-slot rows hidden)
usage: python m3v2_validate2.py"""
import sys, json, pickle
sys.argv = [sys.argv[0], "0"]
import numpy as np, pandas as pd
import t2_m3_gate as G
import m3v2
from t2_model import best_set
from common import REL, NN
from cachelib import load, T2_PANELS

W = G.W
gate = pd.read_csv(W + "m3v2_gate.csv"); parts = pd.read_csv(W + "m3v2_parts.csv"); e3 = pd.read_csv(W + "m3v2_e3.csv"); x1 = pd.read_csv(W + "m3v2_x1.csv")
for D in (gate, parts, e3, x1):
    D["hour"] = [(G.META[w]["t"] % 288) // 12 for w in D.w]
PARTS = [("TTA ongoing", parts, "cur", "tta", "queue_ongoing"), ("cone V3", parts, "cur", "cone", "queue_ongoing"), ("EMA", parts, "ema0", "ema1", None),
         ("onset shift", e3, "cur", "e3", "queue_onset"), ("E3' ongoing", e3, "cur", "e3", "queue_ongoing"),
         ("X1 onset (half)", x1, "base", "mix", "queue_onset"), ("X1 ongoing", x1, "base", "x1", "queue_ongoing")]

# --- V7: propensity weights
spec = json.load(open("t2_specs.json"))["s37"]["spec"]
sp = [(x.split(":")[0], float(x.split(":")[1]) if ":" in x else 1.0) for x in spec.split("+")]
prs = {m: pickle.load(open(W + f"probs_{m}.pkl", "rb")) for m, _ in sp}; tw = sum(w for _, w in sp)
TX = np.load(f"{NN}/nn_test_afdem/X.npy", mmap_mode="r"); TO = np.load(f"{NN}/nn_test_afdem/off.npy")
TP = {str(w): i for i, w in enumerate(np.load(f"{NN}/nn_test_afdem/window_id.npy", allow_pickle=True))}


def cov(p, cond, lq):
    q = p.copy()
    if cond == "queue_onset":
        q[:5] = 0
    s = best_set(q.ravel(), np.ones(q.size, bool)).reshape(q.shape)
    return [float(q[-1].max() if cond == "queue_onset" else q.max()), float(s.sum()), float(lq.sum())]


Xo, Xm, cm = [], [], []
for p in T2_PANELS:
    for r in pd.read_csv(REL / "task2" / p / "private" / "window_index.csv").itertuples():
        pb = sum(w * prs[m][r.window_id] for m, w in sp) / tw; i = TP[r.window_id]; lq = np.asarray(TX[TO[i]:TO[i + 1], G.CI], float) == 1
        Xo.append([r.condition == "queue_onset", int(r.forecast_origin[11:13]), pd.Timestamp(r.date).dayofweek >= 5] + cov(pb, r.condition, lq))
for w in G.WID:
    m = G.META[w]; i = G.POS[w]; lq = np.asarray(G.X[G.OFF[i]:G.OFF[i + 1], G.CI], float) == 1
    Xm.append([m["cond"] == "queue_onset", (m["t"] % 288) // 12, ((5 + m["day"]) % 7) >= 5] + cov(G.NOMEM[w], m["cond"], lq))
Xo, Xm = np.array(Xo, float), np.array(Xm, float)
from sklearn.linear_model import LogisticRegression
def design(X):
    h = X[:, 1]; return np.column_stack([X[:, 0], np.sin(2 * np.pi * h / 24), np.cos(2 * np.pi * h / 24), X[:, 2], X[:, 3], np.log1p(X[:, 4]), np.log1p(X[:, 5])])
clf = LogisticRegression(C=1.0, max_iter=2000).fit(np.vstack([design(Xo), design(Xm)]), np.r_[np.ones(len(Xo)), np.zeros(len(Xm))])
pm = clf.predict_proba(design(Xm))[:, 1]; PW = dict(zip(G.WID, pm / (1 - pm)))


def wmean(D, a, b, mode):
    out = []
    for c in ("queue_onset", "queue_ongoing"):
        g = D[D.cond == c]
        if len(g) == 0 or (g[a] - g[b]).abs().max() < 1e-12:
            continue
        d = (g[b] - g[a]).to_numpy()
        if mode == "raking":
            w, _ = m3v2.weights(g, "private")
        elif mode == "propensity":
            w = g.w.map(PW).to_numpy().astype(float).copy()
            pans = g.panel.to_numpy()
            for pp in np.unique(pans):                                   # keep panels equal
                w[pans == pp] /= w[pans == pp].sum()
        else:
            w = np.ones(len(g))
        out.append(float(np.sum(w * d) / w.sum()))
    return float(np.mean(out)) if out else 0.0


print("V7 weight robustness (T2 difference by weighting scheme)")
for name, D, a, b, c in PARTS:
    g = D if c is None else D[D.cond == c]
    r = {m: wmean(g, a, b, m) for m in ("raking", "propensity", "none")}
    print(f"  {name:16s} raking {r['raking']:+.4f} | propensity {r['propensity']:+.4f} | none {r['none']:+.4f} | same sign {len({np.sign(v) for v in r.values()}) == 1}")

# --- panel heterogeneity (DerSimonian-Laird over panels)
print("\nPanel heterogeneity (DerSimonian-Laird, panel effect = mean difference, SE from windows)")
for name, D, a, b, c in PARTS:
    g = D if c is None else D[D.cond == c]
    eff, var = [], []
    for p, gp in g.groupby("panel"):
        d = (gp[b] - gp[a]).to_numpy()
        if len(d) >= 3:
            eff.append(d.mean()); var.append(max(d.var(ddof=1) / len(d), 1e-10))
    eff, var = np.array(eff), np.array(var); wi = 1 / var; mu_f = np.sum(wi * eff) / wi.sum()
    Q = np.sum(wi * (eff - mu_f) ** 2); dfree = len(eff) - 1; C = wi.sum() - np.sum(wi ** 2) / wi.sum()
    tau2 = max(0.0, (Q - dfree) / C); I2 = max(0.0, (Q - dfree) / Q) if Q > 0 else 0.0
    ws = 1 / (var + tau2); mu = np.sum(ws * eff) / ws.sum(); se = np.sqrt(1 / ws.sum())
    print(f"  {name:16s} random-effects {mu:+.4f} (SE {se:.4f}) | I2 {I2:.0%} | tau {np.sqrt(tau2):.4f} | flag {'ONE-PANEL' if I2 > 0.5 else '-'}")

# --- V8 label noise
near = []; tot = 0
for w in G.WID:
    m = G.META[w]; P, _, _ = m3v2._panel(m["panel"]); t = m["t"]
    V = P.V[t + 1:t + 7]; ok = np.isfinite(V) & P.E[t + 1:t + 7]; cut = P.vcut[None, :]
    nr = ok & (np.abs(V - cut) <= 0.02 * cut); near.append(nr.any()); tot += ok.sum()
print(f"\nV8 label noise: windows with a labelled cell within +-2% of the threshold: {np.mean(near):.2f}")
NW = dict(zip(G.WID, near))
for name, D, a, b, c in PARTS:
    g = D if c is None else D[D.cond == c]; g = g[~g.w.map(NW)]
    s = m3v2.summary(g, a, b, "private", n_boot=300, n_draw=300)
    print(f"  {name:16s} on windows without near-threshold labels (n {len(g)}): {s['t2_weighted']:+.4f}")

# --- V11 fidelity of the simulated blanks
print("\nV11 simulated vs real blanks on evidence slots (share of evidence rows T-30..T+60 min on earlier days that are hidden)")
real = []
for p in T2_PANELS:
    P, _, _ = m3v2._panel(p); ti = {s_: i for i, s_ in enumerate(P.times)}; obs_any = np.isfinite(P.V).any(axis=1)
    for r in pd.read_csv(REL / "task2" / p / "validation" / "window_index.csv").itertuples():
        t = ti[r.forecast_origin]
        for back in (1, 2, 3):
            dd = t - back * 288
            if dd - 6 >= 0:
                real.append(1 - obs_any[dd - 6:dd + 12].mean())             # rows with nothing observed = blank rows
sim = []
for w in G.WID:
    m = G.META[w]
    for k in (1, 2, 3):
        if k > m["day"]:
            continue
        hid = m3v2.sim_blank_rows(m["panel"], m["day"], k, m["t"])
        for back in range(1, k + 1):
            dd = m["t"] - back * 288
            sim.append(np.mean([r_ in hid for r_ in range(dd - 6, dd + 12)]))
print(f"  official March windows (real blanks): {np.mean(real):.3f} of evidence rows hidden (n {len(real)}) | M3 simulated: {np.mean(sim):.3f} (n {len(sim)})")
