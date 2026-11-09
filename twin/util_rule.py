"""Onset rule with same-month flow information (literature B #1 censored capacity / #2 demand look-ahead; premise in
diag_util.py: in the twin's moved maps, links without evidence that carried >= capacity on earlier days without a queue
onset ~3x less often than the NN says, and links whose demand grew since the earlier days more often; flat in the real
train months, so the expected local effect is 0). For links without same-month evidence the x0.25 factor becomes
  0.25 x (beta if u_prev >= 1 else 1) x clip(g_now, 0.5, 2) ** gamma
u_prev: max over the earlier days of the 95th-percentile flow / capacity in slots T-6..T+11 (observed cells, max over
+-2 links); g_now: today's mean flow / capacity in T-6..T-1 over the earlier days' mean in the same slots (1 if unknown).
Threshold 1 fixed by theory (a flow at nominal capacity without breakdown = a censored lower bound on capacity).
Fit (beta, gamma) on one world family, score on the other (R -> T and T -> R), cluster bootstrap over panel x world.
usage: python util_rule.py"""
import sys, pickle, itertools, numpy as np, pandas as pd
sys.path.insert(0, r"C:/Users/user/tfb/twin")
from adjuster import probs, world, dil, bestset
from diag_util import dmax2

GRID = list(itertools.product((1.0, 0.5, 0.25, 0.1), (0.0, 1.0, 2.0, 4.0)))


def score_family(LAB, PR):
    rows = []
    for wid, rec in LAB.items():
        y, el, panel, wd, t, cond, date = rec
        if cond != "queue_onset" or wid not in PR:
            continue
        P, obs, full = world(panel, wd); Y = y.T.astype(bool); EL = el.T.astype(bool)
        fin = np.isfinite(P.V) & ~P.blank
        days = [t - b * 288 for b in (1, 2, 3) if t - b * 288 - 6 >= 0]
        p0 = PR[wid].copy(); p0[:-1] = 0
        def iou(p):
            s = bestset(p); u = ((s | Y) & EL).sum(); return 1.0 if u == 0 else (s & Y & EL).sum() / u
        r = {"cl": panel + wd, "day": int(date[8:10])}
        if not days:
            v = iou(p0); r["cur"] = v; r.update({i: v for i in range(len(GRID))}); rows.append(r); continue
        ev = dil(np.any([obs[dd - 6:dd + 12].any(axis=0) for dd in days], axis=0), 2)
        U = np.where(fin, np.nan_to_num(P.Q) / np.maximum(P.cap[None, :], 1), np.nan)
        with np.errstate(all="ignore"):
            import warnings; warnings.simplefilter("ignore")
            up = dmax2(np.nan_to_num(np.nanmax([np.nanpercentile(U[dd - 6:dd + 12], 95, axis=0) for dd in days], axis=0), nan=0.0))
            g = np.nanmean(U[t - 6:t], axis=0) / np.nanmean([np.nanmean(U[dd - 6:dd], axis=0) for dd in days], axis=0)
        g = np.clip(np.nan_to_num(g, nan=1.0, posinf=1.0), 0.5, 2.0)
        p = p0.copy(); p[-1] = p[-1] * np.where(ev, 1.0, 0.25); r["cur"] = iou(p)
        for i, (beta, gam) in enumerate(GRID):
            p = p0.copy(); p[-1] = np.clip(p[-1] * np.where(ev, 1.0, 0.25 * np.where(up >= 1.0, beta, 1.0) * g ** gam), 0, 1)
            r[i] = iou(p)
        rows.append(r)
    return pd.DataFrame(rows)


def boot(d, cl, rng):
    u = np.unique(cl); idx = {c: np.flatnonzero(cl == c) for c in u}
    return np.std([d[np.concatenate([idx[c] for c in rng.choice(u, len(u))])].mean() for _ in range(500)])


FAM = {"R": (pickle.load(open("E:/tfb_nn/twin_labels_twR.pkl", "rb")), probs("twR_nomem_s0+twR_nomem_s1")),
       "T": (pickle.load(open("E:/tfb_nn/twin_labels.pkl", "rb")), probs("tw2_nomem_s0+tw2_nomem_s1"))}
RES = {f: score_family(*FAM[f]) for f in FAM}
rng = np.random.default_rng(0)
for f, R in RES.items():
    print(f"[{f}] onset windows {len(R)}, current {R.cur.mean():.4f}; top settings (beta, gamma): " +
          ", ".join(f"{GRID[i]} {R[i].mean() - R.cur.mean():+.4f}" for i in np.argsort([-R[i].mean() for i in range(len(GRID))])[:5]))
    for i in (GRID.index((0.25, 0.0)), GRID.index((1.0, 2.0))):
        d = (R[i] - R.cur).to_numpy(); print(f"    {GRID[i]} alone: {d.mean():+.4f} (SE {boot(d, R.cl.to_numpy(), rng):.4f})")
for fit, ev_ in (("R", "T"), ("T", "R")):
    i = max(range(len(GRID)), key=lambda i: RES[fit][i].mean()); R = RES[ev_]; d = (R[i] - R.cur).to_numpy()
    m = ((R.day >= 2) & (R.day <= 4)).to_numpy()
    print(f"fit on {fit} -> {GRID[i]}, scored on {ev_}: {d.mean():+.4f} (SE {boot(d, R.cl.to_numpy(), rng):.4f}); days 2-4 {d[m].mean():+.4f}; days >= 5 {d[R.day.to_numpy() >= 5].mean():+.4f}")
