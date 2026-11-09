"""Premise check for literature B #1/#2 (same-month activation threshold / bottleneck demand look-ahead), before any
rule is built. Onset windows of the twin test worlds T (A-trained NN, tw2_nomem), T+30 step, links WITHOUT same-month
evidence (no queue +-2 links in slots T-6..T+11 on <= 3 earlier days; these are the links the x0.25 rule pushes down):
  u_prev = max over the earlier days of the 95th-percentile flow / capacity (observed cells) in the same slots, +-2 links
  g_now  = today's flow in T-6..T-1 / mean of the earlier days' flow in the same slots at the link (demand growth)
Reports, by bins of u_prev and g_now: share of links, onset rate, mean NN probability, onset / (prob x 0.25) (how far
the rule under- or over-states them). A rule built on u_prev / g_now is worth trying only if this ratio moves strongly
across the bins. Same on the real train months (OOF of the no-memory NN, thinned layer) with --real.
usage: python diag_util.py [--real]"""
import sys, pickle, numpy as np, pandas as pd
sys.path.insert(0, r"C:/Users/user/tfb/twin"); sys.path.insert(0, r"C:/Users/user/tfb/src")
from adjuster import probs, world, dil
from cachelib import load


def dmax2(x):
    e = x.copy()
    for o in (1, 2):
        e[o:] = np.maximum(e[o:], x[:-o]); e[:-o] = np.maximum(e[:-o], x[o:])
    return e


def link_rows(P, qobs, fin, t, n_days_ok, pl, lab):
    """rows for the links without evidence: (u_prev, g_now, prob, label)"""
    U = np.where(fin, np.nan_to_num(P.Q) / np.maximum(P.cap[None, :], 1), np.nan)
    days = [t - b * 288 for b in (1, 2, 3) if n_days_ok(b)]
    if not days:
        return None
    ev = dil(np.any([qobs[dd - 6:dd + 12].any(axis=0) for dd in days], axis=0), 2)
    with np.errstate(all="ignore"):
        up = np.nanmax([np.nanpercentile(U[dd - 6:dd + 12], 95, axis=0) for dd in days], axis=0)
        prev = np.nanmean([np.nanmean(U[dd - 6:dd], axis=0) for dd in days], axis=0)
        g = np.nanmean(U[t - 6:t], axis=0) / prev
    up = dmax2(np.nan_to_num(up, nan=0.0)); m = ~ev
    return np.column_stack([up[m], g[m], pl[m], lab[m]])


rows = []
if "--real" not in sys.argv:
    LT = pickle.load(open("E:/tfb_nn/twin_labels.pkl", "rb")); PA = probs("tw2_nomem_s0+tw2_nomem_s1")
    for wid, rec in LT.items():
        y, el, panel, wd, t, cond, date = rec
        if cond != "queue_onset":
            continue
        P, obs, full = world(panel, wd); fin = np.isfinite(P.V) & ~P.blank
        r = link_rows(P, obs, fin, t, lambda b: t - b * 288 - 6 >= 0, PA[wid][-1], (y.T[-1] & el.T[-1]).astype(bool))
        if r is not None:
            rows.append(r)
else:
    import t2_eval_surprise as S
    from t2_eval_oof import T, load_oof
    oof = load_oof("q25_nomem_s3+q25_nomem_s4+q31_nomem_wp05_s0:2+q31_nomem_attn1_s0:2+q31_nomem_r13_s0+q31_nomem_drop1_s0"); PCR = {}
    for k in S.SEL:
        p = str(T.META["panel"][k]); t = int(T.META["t"][k]); day = pd.Timestamp(str(T.META["date"][k])[:10])
        if p not in PCR:
            P = load(p, "train"); keep = np.random.default_rng(0).random(P.V.shape) > 0.435
            PCR[p] = (P, ((P.V <= P.vcut[None, :]) & np.isfinite(P.V)) & keep, np.isfinite(P.V) & keep)
        P, q, fin = PCR[p]; lab, el = T.LAB[k]
        r = link_rows(P, q, fin, t, lambda b: t - b * 288 - 6 >= 0 and (day - pd.Timedelta(days=b)).month == day.month,
                      oof[k][-1], (lab[-1] & el[-1]).astype(bool))
        if r is not None:
            rows.append(r)
R = pd.DataFrame(np.vstack(rows), columns=["u_prev", "g_now", "p", "y"])
print(f"no-evidence links {len(R)}, onset among them {int(R.y.sum())} ({R.y.mean():.4f}); all-link mean prob {R.p.mean():.4f}")
for col, bins in (("u_prev", [0, .5, .7, .8, .85, .9, .95, 1.0, 9]), ("g_now", [0, .9, .97, 1.0, 1.03, 1.06, 1.1, 1.2, 99])):
    R["b"] = pd.cut(R[col], bins, right=False)
    G = R.groupby("b", observed=True).agg(n=("y", "size"), onset=("y", "sum"), rate=("y", "mean"), prob=("p", "mean"))
    G["share"] = G.n / len(R); G["rate/(0.25p)"] = G.rate / (0.25 * G.prob)
    print(f"--- by {col}"); print(G[["share", "onset", "rate", "prob", "rate/(0.25p)"]].round(4).to_string())
