"""M3 validation V2 + V3 (METHOD_REVIEW_20261022 §1).
V2 LB calibration: the known LB pairs (total-score differences) against the M3 prediction 0.15 x (onset diff) + 0.15 x
(ongoing diff) of the changed conditions, for v1 (m3_gate csv, m3_stats weights) and v2 (m3v2_gate / m3v2_parts csv,
March-mix weights): sign agreement, slope through the origin rho, correlation, leave-one-out prediction error.
V3 balance: standardised mean differences between the v2-weighted M3 windows and the official private (April) windows
for covariates visible before T (hour, weekend, history coverage, base model's max probability, decoded set size,
visible queue size). |SMD| < 0.1 = balanced.
usage: python m3v2_validate.py"""
import sys, json, pickle
sys.argv = [sys.argv[0], "0,1,2,3"]
import numpy as np, pandas as pd
import t2_m3_gate as G
import m3v2, m3_stats
from t2_model import best_set
from common import REL, NN
from cachelib import T2_PANELS

W = G.W
v1 = pd.read_csv(W + "m3_gate_s0-1-2-3.csv"); v2 = pd.read_csv(W + "m3v2_gate.csv"); pv2 = pd.read_csv(W + "m3v2_parts.csv")
cb1 = pd.read_csv(W + "m3_combo.csv")
for D in (v1, v2, pv2, cb1):
    if "hour" not in D:
        D["hour"] = [(G.META[w]["t"] % 288) // 12 for w in D.w]
PAIRS = [  # name, LB total diff, v1 (frame, a, b), v2 (frame, a, b)
    ("memory -> no memory", +0.0032, (v1, "mem_full", "nomem_full"), (v2, "mem", "nomem")),
    ("evidence rule", +0.0032, (v1, "ev_none", "ev_cur"), (v2, "ev_none", "ev_cur")),
    ("far gate", +0.0031, (v1, "gate_off", "gate_on"), (v2, "gate_off", "gate_on")),
    ("sim ongoing (s60->s70)", +0.0035, (v1, "nomem_full", "sim_full"), (v2, "nomem", "sim")),
    ("B1 skip (s70->s72)", -0.0021, (v1, "ev_cur", "b1_skip"), (v2, "ev_cur", "skip")),
    ("negative control (s49)", -0.0051, (v1, "nomem_full", "neg_full"), (v2, "nomem", "neg")),
    ("s75 TTA + cone", +0.00042, (cb1, "cur", "March: TTA+cone"), (pv2, "cur", "tta_cone")),
]


def pred(D, a, b, version):
    g = D.dropna(subset=[a, b])
    out = 0.0
    for c in ("queue_onset", "queue_ongoing"):
        gc = g[g.cond == c]
        if len(gc) == 0 or (gc[a] - gc[b]).abs().max() < 1e-12:
            continue
        s = m3_stats.summary(gc, a, b, n_boot=50, n_draw=50) if version == 1 else m3v2.summary(gc, a, b, "validation", n_boot=50, n_draw=50)
        out += 0.15 * s["per_cond"][c]["weighted"]
    return out


rows = []
for name, lb, p1, p2 in PAIRS:
    rows.append(dict(pair=name, lb=lb, m3_v1=pred(*p1, 1), m3_v2=pred(*p2, 2)))
R = pd.DataFrame(rows)
print("V2 LB calibration (predicted LB difference = 0.15 x changed-condition M3 differences)")
print(R.round(5).to_string(index=False))
for v in ("m3_v1", "m3_v2"):
    x, y = R[v].to_numpy(), R.lb.to_numpy(); rho = float(x @ y / (x @ x)); res = y - rho * x
    loo = [y[i] - (np.delete(x, i) @ np.delete(y, i) / (np.delete(x, i) @ np.delete(x, i))) * x[i] for i in range(len(x))]
    print(f"  {v}: signs {int(np.sum(np.sign(x) == np.sign(y)))}/{len(x)}, slope rho {rho:.2f}, corr {np.corrcoef(x, y)[0, 1]:.2f}, residual sd {res.std():.4f}, LOO error sd {np.std(loo):.4f}")

# V3 balance
spec = json.load(open("t2_specs.json"))["s37"]["spec"]
parts = [(x.split(":")[0], float(x.split(":")[1]) if ":" in x else 1.0) for x in spec.split("+")]
prs = {m: pickle.load(open(W + f"probs_{m}.pkl", "rb")) for m, _ in parts}; tw = sum(w for _, w in parts)
TX = np.load(f"{NN}/nn_test_afdem/X.npy", mmap_mode="r"); TO = np.load(f"{NN}/nn_test_afdem/off.npy")
TP = {str(w): i for i, w in enumerate(np.load(f"{NN}/nn_test_afdem/window_id.npy", allow_pickle=True))}


def cov_feats(p, cond, lq):
    q = p.copy()
    if cond == "queue_onset":
        q[:5] = 0
    s = best_set(q.ravel(), np.ones(q.size, bool)).reshape(q.shape)
    return dict(maxp=float(q[-1].max() if cond == "queue_onset" else q.max()), setsize=float(s.sum()), lq=float(lq.sum()))


off = []
for p in T2_PANELS:
    for r in pd.read_csv(REL / "task2" / p / "private" / "window_index.csv").itertuples():
        pb = sum(w * prs[m][r.window_id] for m, w in parts) / tw; i = TP[r.window_id]; lq = np.asarray(TX[TO[i]:TO[i + 1], G.CI], float) == 1
        off.append(dict(cond=r.condition, hour=int(r.forecast_origin[11:13]), we=float(pd.Timestamp(r.date).dayofweek >= 5), cov=r.history_coverage, **cov_feats(pb, r.condition, lq)))
O = pd.DataFrame(off)
m3 = []
for w in G.WID:
    m = G.META[w]; i = G.POS[w]; lq = np.asarray(G.X[G.OFF[i]:G.OFF[i + 1], G.CI], float) == 1
    m3.append(dict(w=w, cond=m["cond"], panel=m["panel"], day=m["day"], hour=(m["t"] % 288) // 12, we=float(((5 + m["day"]) % 7) >= 5), cov=m["cov"], **cov_feats(G.NOMEM[w], m["cond"], lq)))
M = pd.DataFrame(m3); wts, _ = m3v2.weights(M, "private")
print("\nV3 balance: standardised mean differences, v2-weighted M3 vs official April windows (and unweighted M3)")
for c in ("queue_onset", "queue_ongoing"):
    mc = (M.cond == c).to_numpy(); oc = O[O.cond == c]; out = []
    for v in ("hour", "we", "cov", "maxp", "setsize", "lq"):
        x = M.loc[mc, v].to_numpy(float); wv = wts[mc] / wts[mc].sum(); mw = float(wv @ x); mu = float(x.mean()); mo = float(oc[v].mean())
        sd = np.sqrt((x.var() + oc[v].var()) / 2) + 1e-12
        out.append(f"{v} {(mw - mo) / sd:+.2f} (unweighted {(mu - mo) / sd:+.2f})")
    print(f"  {c}: " + " | ".join(out))
