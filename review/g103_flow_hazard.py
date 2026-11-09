"""G103 (10/24, X lanes 1b + 2 + 3, LITERATURE_SCAN_20261024 §1 / §2-1 / §2-4): a flow-only breakdown hazard for the onset site.
An onset window has no queued cell in rows T-12..T+5 and a queue at T+6 (t2_windows), so the site question is WHERE the
corridor's first queue forms at T+6, and the flows of rows T+3..T+5 are still free of it (the oracle input below).
Model (train months only; nothing of March is used to fit it):
  head of a new queue at link j in row t: queued at t, j+1 not queued at t, links j-3..j+3 free in rows t-6..t-1;
  at risk: the same freedom and j not queued at t (or the head). q = 15-min mean flow per lane at j (rows t-3..t-1).
  per-row hazard h_j(q) = 1 - exp(-(q / lambda_j)^gamma_c) (Brilon's Weibull capacity), classes c = lanes {<=2, 3, 4, >=5}
  x on-ramp attached x lane drop at j+1; lambda_j = lambda_c exp(u_j), u_j ~ N(0, s^2), s from the spread of the sites
  with >= 50 events (Mikolasek 2026: below that a site's capacity is not identifiable), MAP u_j for every link.
Joint sampler of the T+30 queued set: number of heads and block lengths ~ the train onset events (corridor free 85 min,
queue at the row), heads ~ h/(1-h) without replacement (Gumbel top-k).
Outputs on the M3 v3 dev onset windows (hold-out excluded), T+30 cells as the instrument scores them:
  (i)   model-world expected IoU of the MBR set = the flow-only 'ceiling' (oracle flows)
  (ii)  realized IoU of the MBR set / of best_set on the sampler's marginals, oracle flows and as-of-T flows (rows T-2..T)
  (iii) information test: the final A's T+30 p mixed with the as-of-T marginals, b in {0.1, 0.2, 0.3} (k = 3), through the
        production decoder (evidence rule, k expectation) -> work/t2/g103_haz_mix.csv for judge.py --cond queue_onset
Pre-declared wording: (i) within error of the current IoU -> "flow information exhausted under this model", not a stochastic
ceiling of onset.
usage: TFB_M3VER=v3 python g103_flow_hazard.py"""
import os, sys, pickle, time
os.environ.setdefault("TFB_M3VER", "v3")
sys.path.insert(0, r"C:/Users/user/tfb/src"); argv0 = sys.argv[0]
import numpy as np, pandas as pd, lightgbm as lgb
from scipy.ndimage import maximum_filter1d
from scipy.optimize import minimize
from cachelib import load, T2_PANELS
from t2_windows import queue_matrix
import t2_phys_onset as PO
sys.argv = [argv0, "0,1,2,3"]
import t2_m3_gate as G, t2_m3v2_gate as V, m3v2
from t2_model import best_set
from common import NN

W = G.W; VER = G.VER; PRE = G.PRE; t0 = time.time()
QB = np.arange(0, 3025, 25.0); QC = QB[:-1] + 12.5            # per-lane flow bins (vph)
rng = np.random.default_rng(103)


def cls(P):
    lb = np.digitize(P.lanes, [2.5, 3.5, 4.5]); drop = np.r_[P.lanes[1:] < P.lanes[:-1], False]
    return lb * 4 + P.on_attached.astype(int) * 2 + drop.astype(int)


def q15(P, r0, r1):
    """mean flow per lane over rows r0..r1-1 (NaN where nothing observed)"""
    with np.errstate(all="ignore"):
        return np.nanmean(P.Q[max(r0, 0):r1], axis=0) / P.lanes


def runs(row):
    """(head, length) of the contiguous queued runs of a boolean link row (head = most downstream)"""
    d = np.diff(np.r_[0, row.astype(int), 0]); s, e = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
    return [(b - 1, b - a) for a, b in zip(s, e)]


# ---- 1. train months: per-link histograms of at-risk rows and heads by flow bin; onset events' head counts and lengths
HIST, NHEAD, LENS = {}, [], []
PANELS = T2_PANELS[:int(os.environ.get('G103_NP', len(T2_PANELS)))]   # smoke test: the first panels only
for p in PANELS:
    P = load(p, "train"); Qm = queue_matrix(P); T, n = Qm.shape
    near = maximum_filter1d(Qm.astype(np.uint8), 7, axis=1)              # queue within +-3 links
    cs = np.vstack([np.zeros((1, n), np.int32), np.cumsum(near, axis=0, dtype=np.int32)])
    free = np.zeros((T, n), bool); free[6:] = (cs[6:T] - cs[0:T - 6]) == 0  # rows t-6..t-1 free around j
    head = Qm & ~np.c_[Qm[:, 1:], np.zeros((T, 1), bool)]
    q = np.full((T, n), np.nan, np.float32)
    with np.errstate(all="ignore"):
        Qf = np.where(np.isfinite(P.Q), P.Q, 0.0); Qn = np.isfinite(P.Q).astype(np.float32)
        c1 = np.vstack([np.zeros((1, n)), np.cumsum(Qf, 0)]); c2 = np.vstack([np.zeros((1, n)), np.cumsum(Qn, 0)])
        q[3:] = ((c1[3:T] - c1[0:T - 3]) / (c2[3:T] - c2[0:T - 3])) / P.lanes
    risk = free & (~Qm | head) & np.isfinite(q) & P.E; ev = risk & head
    b = np.clip(np.digitize(np.nan_to_num(q), QB) - 1, 0, len(QC) - 1)
    nr = np.zeros((n, len(QC))); ne = np.zeros((n, len(QC)))
    for j in range(n):
        nr[j] = np.bincount(b[risk[:, j], j], minlength=len(QC)); ne[j] = np.bincount(b[ev[:, j], j], minlength=len(QC))
    HIST[p] = (nr, ne, cls(P))
    anyq = Qm.any(axis=1); ca = np.r_[0, np.cumsum(anyq)]
    for t in np.flatnonzero(anyq[18:]) + 18:                               # corridor free in rows t-18..t-1 (T-12..T+5)
        if ca[t] - ca[t - 18] == 0:
            r = runs(Qm[t]); NHEAD.append(len(r)); LENS += [L for _, L in r]
    print(p, "links", n, "heads", int(ne.sum()), "at-risk rows", int(nr.sum()), "onset events", len(NHEAD), f"{time.time() - t0:.0f}s", flush=True)
    del P, Qm, near, cs, free, head, q, risk, ev, b, Qf, Qn, c1, c2; load.cache_clear()
NH = np.bincount(NHEAD)[1:] / len(NHEAD); LD = np.bincount(LENS)[1:] / len(LENS)
print("train onset events", len(NHEAD), "| heads per event", np.round(NH[:4], 3), "| block length 1..6", np.round(LD[:6], 3), flush=True)


def ll(lam, gam, nr, ne):
    F = np.clip(1 - np.exp(-(QC / lam) ** gam), 1e-12, 1 - 1e-12)
    return (ne * np.log(F) + (nr - ne) * np.log1p(-F)).sum(-1)


# ---- 2. class Weibull (lambda_c, gamma_c), site lambda for >= 50 events, MAP u_j for every link
NR = np.vstack([h[0] for h in HIST.values()]); NE = np.vstack([h[1] for h in HIST.values()]); CL = np.concatenate([h[2] for h in HIST.values()])
PAR = {}
for c in np.unique(CL):
    nr, ne = NR[CL == c].sum(0), NE[CL == c].sum(0)
    if ne.sum() < 20:
        continue
    o = minimize(lambda x: -ll(np.exp(x[0]), np.exp(x[1]), nr, ne), [np.log(2600.0), np.log(8.0)], method="Nelder-Mead", options=dict(maxiter=2000, xatol=1e-4, fatol=1e-4))
    PAR[c] = (float(np.exp(o.x[0])), float(np.exp(o.x[1])), int(ne.sum()))
gl = minimize(lambda x: -ll(np.exp(x[0]), np.exp(x[1]), NR.sum(0), NE.sum(0)), [np.log(2600.0), np.log(8.0)], method="Nelder-Mead").x
GLOB = (float(np.exp(gl[0])), float(np.exp(gl[1])))
lc = lambda c: PAR.get(c, GLOB)[:2]
UG = np.linspace(-1.5, 1.5, 301)
big = [j for j in range(len(CL)) if NE[j].sum() >= 50]
ubig = [UG[np.argmax([ll(lc(CL[j])[0] * np.exp(u), lc(CL[j])[1], NR[j], NE[j]) for u in UG])] for j in big]
S = float(np.std(ubig)) if len(ubig) >= 3 else 0.1
U = np.zeros(len(CL))
for j in range(len(CL)):
    lam, gam = lc(CL[j]); U[j] = UG[np.argmax(ll(lam * np.exp(UG)[:, None], gam, NR[j][None, :], NE[j][None, :]) - UG ** 2 / (2 * S ** 2))]
print("classes", {int(c): (round(v[0]), round(v[1], 2), v[2]) for c, v in PAR.items()}, "| global", np.round(GLOB, 2),
      f"| sites >= 50 events {len(big)}, s = {S:.3f} | u range {U.min():+.2f}..{U.max():+.2f}", flush=True)
LAMJ, GAMJ, k0 = {}, {}, 0
for p, (nr, _, c) in HIST.items():
    LAMJ[p] = np.array([lc(x)[0] for x in c]) * np.exp(U[k0:k0 + len(c)]); GAMJ[p] = np.array([lc(x)[1] for x in c]); k0 += len(c)
pickle.dump(dict(LAMJ=LAMJ, GAMJ=GAMJ, NH=NH, LD=LD, PAR=PAR, S=S), open(W + "g103_hazard.pkl", "wb"))


# ---- 3. the sampler and the decoders on the M3 dev onset windows
def sample(h, ns=2000):
    n = len(h); w = np.log(np.clip(h / np.clip(1 - h, 1e-9, None), 1e-30, None))
    key = w[None, :] + rng.gumbel(size=(ns, n)); order = np.argsort(-key, axis=1)
    nh = rng.choice(np.arange(1, len(NH) + 1), size=ns, p=NH); Y = np.zeros((ns, n), bool)
    Ls = rng.choice(np.arange(1, len(LD) + 1), size=(ns, int(nh.max())), p=LD)
    for s in range(ns):
        for hd, L in zip(order[s, :nh[s]], Ls[s]):
            Y[s, max(0, hd - L + 1):hd + 1] = True
    return Y


def eiou(c, Y, mk):
    Yk = Y[:, mk]; ck = c[mk]; i = Yk[:, ck].sum(1); u = Yk.sum(1) + ck.sum() - i
    return np.where(u == 0, 1.0, i / np.maximum(u, 1))


def mbr(Y, h, mk):
    """candidate sets: best_set on the marginals; blocks of length 1..5 at the 6 most likely heads; unions of two of the top
    4 heads' blocks (length 1..3) -> the one with the largest expected IoU over the samples"""
    pm = Y.mean(0); C = [best_set(pm, np.ones(len(pm), bool))]; top = np.argsort(-h)[:6]; n = len(h)
    blk = lambda hd, L: np.isin(np.arange(n), np.arange(max(0, hd - L + 1), hd + 1))
    C += [blk(hd, L) for hd in top for L in range(1, 6)]
    C += [blk(a, La) | blk(b_, Lb) for i, a in enumerate(top[:4]) for b_ in top[i + 1:4] for La in (1, 2, 3) for Lb in (1, 2, 3)]
    e = [eiou(c, Y, mk).mean() for c in C]; k = int(np.argmax(e))
    return C[k], float(e[k]), C[0], float(e[0]), pm


def hz(lam, gam, q):
    return 1 - np.exp(-(np.nan_to_num(q, nan=0.0).clip(0) / lam) ** gam)


CF5 = (("attnglob", 2.0), ("attnwp05", 2.0), ("attn1", 2.0), ("drop1", 1.0), ("r13", 1.0))
ON = [(f"q24f_{c}_s{s}", w) for s in range(8) for c, w in CF5]; PRO = {t: pickle.load(open(W + f"probs_{PRE}__{t}.pkl", "rb")) for t, _ in ON}
X = np.load(f"{NN}/nn_test_m3{VER}/X.npy", mmap_mode="r"); OFF = np.load(f"{NN}/nn_test_m3{VER}/off.npy")
phys = lgb.Booster(model_file=W + "physon_base.txt"); ev = PO.month_events("validation", PO.rarity_table())
iou = lambda c, y, mk: 1.0 if not ((c | y) & mk).any() else float((c & y & mk).sum() / ((c | y) & mk).sum())
rows, cols = [], {v: {} for v in ("A", "haz10", "haz20", "haz30")}
for i, w in enumerate(G.WID):
    m = G.META[w]
    if m["cond"] != "queue_onset" or m3v2.is_holdout(np.array([m["day"]]))[0] or m["panel"] not in PANELS:
        continue
    P, _, _ = m3v2._panel(m["panel"]); t = m["t"]; mk = m["m"].T.astype(bool)[-1]; y = m["y"].T.astype(bool)[-1]
    lam, gam = LAMJ[m["panel"]], GAMJ[m["panel"]]
    qa = q15(P, t - 2, t + 1); qo = q15(P, t + 3, t + 6); qo = np.where(np.isfinite(qo), qo, qa)
    r = dict(w=w, panel=m["panel"], day=m["day"], hour=int(t % 288) // 12)
    for tag, qq in (("orc", qo), ("asof", qa)):
        h = hz(lam, gam, qq); Y = sample(h); cm, em, cb, eb, pm = mbr(Y, h, mk)
        r.update({f"{tag}_ceil": em, f"{tag}_mbr": iou(cm, y, mk), f"{tag}_bs": iou(cb, y, mk), f"{tag}_bs_exp": eb,
                  f"{tag}_mass_true": float(pm[y & mk].sum() / max(pm[mk].sum(), 1e-9)), f"{tag}_rank_true": float(np.mean([(h > h[j]).sum() for j in np.flatnonzero(y & mk)])) if (y & mk).any() else np.nan})
        if tag == "asof":
            PMA = pm
    Xw = np.asarray(X[OFF[i]:OFF[i + 1]]); p = sum(wt * PRO[tg][w] for tg, wt in ON) / sum(wt for _, wt in ON); a = PO.adaptive_a(ev, m["day"])
    p = p.copy(); p[5] = (1 - a) * p[5] + a * phys.predict(PO.feats(Xw))
    r["A_mass_true"] = float(p[5][y & mk].sum() / max(p[5][mk].sum(), 1e-9))
    for v, bb in (("A", 0.0), ("haz10", 0.1), ("haz20", 0.2), ("haz30", 0.3)):
        pv = p.copy(); pv[5] = (1 - bb) * p[5] + bb * PMA
        cols[v][w] = sum(pr * G.iou(V.onset_pred(pv, m, "cur", k), m) for k, pr in m3v2.kdist(m["day"]).items())
    rows.append(r)
D = pd.DataFrame(rows); D["A"] = D.w.map(cols["A"]); D.to_csv(W + "g103_flow_hazard.csv", index=False)
V.frame(cols).to_csv(W + "g103_haz_mix.csv", index=False)
print(f"M3 {VER} dev onset windows {len(D)} ({time.time() - t0:.0f}s): current final A realized {D.A.mean():.4f}")
for tag in ("orc", "asof"):
    print(f"  {tag:4s}: model-world ceiling (MBR expected) {D[tag + '_ceil'].mean():.4f} | realized MBR {D[tag + '_mbr'].mean():.4f} | "
          f"realized best_set {D[tag + '_bs'].mean():.4f} (expected {D[tag + '_bs_exp'].mean():.4f}) | mass on true cells {D[tag + '_mass_true'].mean():.3f} "
          f"(final A {D.A_mass_true.mean():.3f}) | links ranked above a true cell {D[tag + '_rank_true'].mean():.1f}")
unc = D.w.isin(pd.read_csv(W + "g102_onset_selfcal.csv").query("c_sel < 0.8").w)
print(f"  uncertain windows (G102 top p < 0.8): {unc.sum()} | final A {D.A[unc].mean():.4f} | mixes " +
      " ".join(f"{v} {D.w[unc].map(cols[v]).mean():.4f}" for v in ("haz10", "haz20", "haz30")))
