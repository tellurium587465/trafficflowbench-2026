"""G54: two onset tests from the G52 loss map (STATE.md 10/22 pre-registration; M3, NOMEM blend, current post-processing).
Link frequency f (train months only): share of the train months' selector-like onset windows of the same panel x
weekend x 2-hour slot in which the link is queued at T+30 (eligible cells); rare = f <= 0.1.
  T1 rare-site exemption: the evidence rule (x0.25 on links unseen on earlier days) is not applied to rare links.
  T2 structural law (G48) on M3: p_law per block = sigmoid(a + b log u + c log u0), u = f_s(T) mu(slot T+6) / mu(slot T-2)
     / C_s, u0 = f_s(T) / C_s; blocks, capacity C and the demand shape mu from the train months (cache, all rows), the
     block flow f_s(T) from the March rows T-3..T-1 (masked layer, read through cachelib.asof). (a, b, c) fitted on the
     train months' selector-like onset windows with G48's fold-out statistics. For rare links without evidence (where
     x0.25 would apply): logit p' = logit p + gamma logit p_law, gamma in {0, 0.1, 0.25, 0.5, 1} chosen leave-one-panel-out
     (gamma = 0 is T1).
Scored at T+30 on observed cells (as the M3 gate); raw / production-weighted differences, cluster SE, P(B>A).
usage: python g54_m3_onset_rare.py"""
import sys
sys.path.insert(0, r"C:/Users/user/tfb/src"); sys.path.insert(0, r"C:/Users/user/tfb/twin")
sys.argv = [sys.argv[0], "0,1,2,3"]
import numpy as np, pandas as pd
from scipy.optimize import minimize
import t2_m3_gate as G
from m3_stats import summary, fmt, MARCH_FIRST_DOW
from t2_eval_oof import T
from cachelib import load, asof
from t2_model import best_set
from onset_shape3_lib import blocks

lg = lambda x: np.log(np.clip(x, 1e-5, 1 - 1e-5) / (1 - np.clip(x, 1e-5, 1 - 1e-5)))
sg = lambda z: 1 / (1 + np.exp(-z))
FOLD = {**{m: 0 for m in ("2030-06", "2030-07", "2030-08")}, **{m: 1 for m in ("2030-09", "2030-10", "2030-11")},
        **{m: 2 for m in ("2030-12", "2031-01", "2031-02")}}
SELON = [k for k in range(T.NWIN) if T.META["is_sel"][k] and str(T.META["cond"][k]) == "queue_onset"]
tkey = lambda p, we, t: (str(p), bool(we), (int(t) % 288) // 24)

# link frequency f from the train months
CNT, DEN = {}, {}
for k in SELON:
    lab, el = T.LAB[k]; y = (lab[5] & el[5]).astype(float)
    key = tkey(T.META["panel"][k], pd.Timestamp(str(T.META["date"][k])[:10]).dayofweek >= 5, T.META["t"][k])
    CNT[key] = CNT.get(key, 0) + y; DEN[key] = DEN.get(key, 0) + 1
WON = [w for w in G.WID if G.META[w]["cond"] == "queue_onset"]
def mkey(m): return tkey(m["panel"], ((MARCH_FIRST_DOW + m["day"]) % 7) >= 5, m["t"])
RARE = {w: (CNT[mkey(G.META[w])] / DEN[mkey(G.META[w])] if mkey(G.META[w]) in DEN else np.zeros(len(G.META[w]["y"]))) <= 0.1 for w in WON}
print("M3 onset windows", len(WON), "| slots without train windows", sum(mkey(G.META[w]) not in DEN for w in WON),
      "| mean share of rare links", round(float(np.mean([RARE[w].mean() for w in WON])), 3), flush=True)

# structural law: train fit (fold-out statistics, as G48) and M3 features (all train rows)
def block_stats(P, rows, slot, we):
    V, Q = P.V, P.Q; bid = blocks(V, P, rows); nb = bid.max() + 1; cong = (V <= P.vcut[None, :]) & np.isfinite(V)
    fb = np.stack([np.nanmean(Q[:, bid == j], axis=1) for j in range(nb)], axis=1)
    cb = np.stack([cong[:, bid == j].any(axis=1) for j in range(nb)], axis=1)
    C = np.array([np.nanmedian(fb[rows & cb[:, j], j]) if (rows & cb[:, j]).sum() >= 30 else np.nan for j in range(nb)])
    fbm = np.where(cb, np.nan, fb); mu = np.full((2, 288, nb), np.nan)
    for w_ in (0, 1):
        for s in range(288):
            sel = rows & (we == w_) & (slot == s)
            if sel.sum() > 3:
                mu[w_, s] = np.nanmean(fbm[sel], axis=0)
    return bid, fb, C, mu


def feats(ft, mu, C, w_, s0):
    u = ft * mu[w_, min(s0 + 6, 287)] / mu[w_, s0 - 2] / C; u0 = ft / C
    cl = lambda x: np.log(np.clip(np.nan_to_num(x, nan=0.05), 0.05, 3))
    return cl(u), cl(u0), np.isfinite(C)


XS, YS, LAW = [], [], {}
for p in sorted({G.META[w]["panel"] for w in WON}):
    P = load(p, "train"); D_ = P.T // 288; slot = np.arange(P.T) % 288
    we = np.repeat(np.array([pd.Timestamp(str(x)).dayofweek >= 5 for x in P.day[:D_]]), 288)[:P.T]
    fold_row = np.array([FOLD.get(str(t)[:7], -1) for t in P.times])
    kp = [k for k in SELON if str(T.META["panel"][k]) == p]
    for f in range(3):
        rows = fold_row != f; bid, fb, C, mu = block_stats(P, rows, slot, we)
        for k in kp:
            if FOLD[str(T.META["date"][k])[:7]] != f:
                continue
            t = int(T.META["t"][k]); lu, lu0, cand = feats(np.nanmean(fb[t - 3:t], axis=0), mu, C, int(we[t]), slot[t])
            lab, el = T.LAB[k]; y = lab[5].astype(bool) & el[5].astype(bool)
            yb = np.array([y[bid == j].any() for j in range(bid.max() + 1)])
            XS.append(np.column_stack([np.ones(cand.sum()), lu[cand], lu0[cand]])); YS.append(yb[cand].astype(float))
    bid, fb, C, mu = block_stats(P, np.ones(P.T, bool), slot, we)
    Pv = load(p, "validation")
    for w in WON:
        m = G.META[w]
        if m["panel"] != p:
            continue
        t = m["t"]; Pt = asof(Pv, t); nb = bid.max() + 1
        ft = np.array([np.nanmean(Pt.Q[t - 3:t][:, bid == j]) if np.isfinite(Pt.Q[t - 3:t][:, bid == j]).any() else np.nan for j in range(nb)])
        LAW[w] = (feats(ft, mu, C, int(((MARCH_FIRST_DOW + m["day"]) % 7) >= 5), t % 288), bid)
    print(p, "law features done", flush=True)
    load.cache_clear()
A, Y = np.vstack(XS), np.concatenate(YS)
fobj = lambda w: (np.mean(np.logaddexp(0, A @ w) - Y * (A @ w)) + 1e-7 * (w @ w), A.T @ (sg(A @ w) - Y) / len(Y) + 2e-7 * w)
WL = minimize(fobj, np.zeros(3), jac=True, method="L-BFGS-B").x
print("law weights (a, b, c) =", np.round(WL, 3).tolist(), "| train blocks", len(Y), "positives", int(Y.sum()), flush=True)


def plaw(w):
    (lu, lu0, cand), bid = LAW[w]
    return np.clip(np.where(cand, sg(WL[0] + WL[1] * lu + WL[2] * lu0), 1e-4)[bid], 1e-5, 1 - 1e-5)


def onset(p_, m, w, mode, gamma=0.0):
    pb = p_.copy(); s, nd = G.seen_rows(m["panel"], m["t"], m["k"], -6, 12)
    if nd:
        ev = G.dil(s, 2); fac = np.where(ev, 1.0, 0.25)
        if mode in ("t1", "t2"):
            z = RARE[w] & ~ev; fac = np.where(z, 1.0, fac)
        pb[-1] = pb[-1] * fac
        if mode == "t2" and gamma:
            pb[-1] = np.where(z, sg(lg(pb[-1]) + gamma * lg(plaw(w))), pb[-1])
    return best_set(pb.ravel(), np.ones(pb.size, bool)).reshape(pb.shape)


GAM = [0.0, 0.1, 0.25, 0.5, 1.0]
rows, fpfn = [], {a: np.zeros(4, int) for a in ("cur", "t1")}
SC = {g: {} for g in GAM}
for w in WON:
    m = G.META[w]; mk = m["m"].T.astype(bool)[-1]; y = m["y"].T.astype(bool)[-1] & mk
    cur = onset(G.NOMEM[w], m, w, "cur"); assert (cur == G.onset_pred(G.NOMEM[w], m, "cur")).all(), "current rule not reproduced"
    t1 = onset(G.NOMEM[w], m, w, "t1")
    for g in GAM:
        SC[g][w] = G.iou(onset(G.NOMEM[w], m, w, "t2", g), m)
    for a, sel in (("cur", cur), ("t1", t1)):
        s = sel[-1] & mk; r = RARE[w]
        fpfn[a] += np.array([(s & ~y & r).sum(), (s & ~y & ~r).sum(), (y & ~s & r).sum(), (y & ~s & ~r).sum()])
    rows.append(dict(w=w, cond="queue_onset", panel=m["panel"], day=m["day"], cur=G.iou(cur, m), t1=G.iou(t1, m)))
D = pd.DataFrame(rows)
assert abs(D.t1.mean() - np.mean([SC[0.0][w] for w in D.w])) < 1e-12, "gamma 0 differs from T1"
chosen = {}
for ph in sorted(D.panel.unique()):
    tr = D[D.panel != ph].w
    chosen[ph] = max(GAM, key=lambda g: np.mean([SC[g][w] for w in tr]))
D["t2"] = [SC[chosen[p]][w] for w, p in zip(D.w, D.panel)]
D.to_csv(G.W + "g54_m3_onset_rare.csv", index=False)
print("gamma chosen per held-out panel:", chosen, "| in-sample mean IoU by gamma:", {g: round(float(np.mean(list(SC[g].values()))), 4) for g in GAM})
for a in ("cur", "t1"):
    v = fpfn[a]; print(f"{a}: FP cells rare {v[0]} / common {v[1]} | FN cells rare {v[2]} / common {v[3]}")
print(fmt("T1 rare-site exemption vs current", summary(D, "cur", "t1")))
print(fmt("T2 law prior (nested gamma) vs current", summary(D, "cur", "t2")))
print(fmt("T2 vs T1", summary(D, "t1", "t2")))
