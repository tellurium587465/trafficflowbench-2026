"""Which second final (hedge) adds most? Kaggle ranks a team by the better of its two selected submissions on the
private month, so the hedge should have a similar expected score but different predictions (literature C).
Twin test worlds T; per-window IoU for each candidate; a 'month' = 5 onset + 5 ongoing windows per panel from one
world (as the private month); S = 0.30 x mean over panels of (onset + ongoing)/2 (T2 part only, the rest is equal).
Reports E[S_main], E[S_hedge] - E[S_main], E[max(main, hedge)] - E[S_main] over 5000 simulated months.
Candidates (onset model = A-only NN, ongoing main = R24+R48 sim-trained NN, ongoing alt = A-only NN):
  main : onset B 0.25           | ongoing sim  (OGB 0.85, gate + contig)
  H1   : onset B 0.25           | ongoing A-only            (= s60, the current hedge)
  H2   : onset prior-shift (0.5, 1) | ongoing sim
  H3   : onset B 0.5            | ongoing sim
  H4   : onset prior-shift      | ongoing A-only
usage: python hedge.py [a,b a,b ...]   (extra prior-shift settings, each scored as hedge 'PS<a>_<b>' with ongoing sim)"""
import sys, pickle, numpy as np, pandas as pd
sys.path.insert(0, r"C:/Users/user/tfb/twin")
from adjuster import probs, bestset, world, dil, gate, rule_probs
from prior_shift import train_freq

LT = pickle.load(open("E:/tfb_nn/twin_labels.pkl", "rb"))
PA = probs("tw2_nomem_s0+tw2_nomem_s1"); PS = probs("tw2ar24_nomem_s0+tw2ar24_nomem_s1+tw2ar48t2_nomem_s0+tw2ar48t2_nomem_s1")


def iou(s, Y, EL):
    u = ((s | Y) & EL).sum(); return 1.0 if u == 0 else (s & Y & EL).sum() / u


XPS = [tuple(float(x) for x in v.split(",")) for v in sys.argv[1:]]
rows = []
for wid, rec in LT.items():
    y, el, panel, wd, t, cond, date = rec
    P, obs, full = world(panel, wd); Y = y.T.astype(bool); EL = el.T.astype(bool); r = dict(panel=panel, wd=wd, cond=cond)
    if cond == "queue_ongoing":
        fut = Y & EL; pp = np.broadcast_to(full[t], fut.shape) & EL; u = (pp | fut).sum()
        if u and (pp & fut).sum() / u > 0.9:
            continue
        r["sim"] = iou(gate(bestset(rule_probs(rec, PS[wid])), rec), Y, EL)
        r["aonly"] = iou(gate(bestset(rule_probs(rec, PA[wid])), rec), Y, EL)
    else:
        days = []
        for b in (1, 2, 3):
            dd = t - b * 288
            if dd - 6 >= 0:
                days.append(dil(obs[dd - 6:dd + 12].any(axis=0), 2))
        n = len(days); ev = np.any(days, axis=0) if n else np.zeros(P.n, bool); k = np.sum(days, axis=0) if n else 0
        p0 = PA[wid].copy(); p0[:-1] = 0
        for name, B in (("b025", 0.25), ("b05", 0.5)):
            p = p0.copy()
            if n:
                p[-1] = p[-1] * np.where(ev, 1.0, B)
            r[name] = iou(bestset(p), Y, EL)
        p = p0.copy()
        if n:
            f = np.clip(train_freq(panel)[int(pd.Timestamp(date).dayofweek >= 5), t % 288], 0.02, 1.0)
            fhat = (k + 0.5 * f) / (n + 0.5); p[-1] = np.clip(p[-1] * np.clip(fhat / f, 0.02, 50), 0, 1)
        r["ps"] = iou(bestset(p), Y, EL)
        for a_, b_ in XPS:
            p = p0.copy()
            if n:
                p[-1] = np.clip(p[-1] * np.clip((k + a_ * f) / (n + a_) / f, 0.02, 50) ** b_, 0, 1)
            r[f"ps{a_}_{b_}"] = iou(bestset(p), Y, EL)
    rows.append(r)
R = pd.DataFrame(rows)
CAND = {"main": ("b025", "sim"), "H1": ("b025", "aonly"), "H2": ("ps", "sim"), "H3": ("b05", "sim"), "H4": ("ps", "aonly")}
CAND.update({f"PS{a_}_{b_}": (f"ps{a_}_{b_}", "sim") for a_, b_ in XPS})
rng = np.random.default_rng(0); res = {c: [] for c in CAND}
groups = {(p, w): g for (p, w), g in R.groupby(["panel", "wd"])}
panels = sorted(R.panel.unique())
for _ in range(5000):
    on_rows, og_rows = [], []
    per_panel = {c: [] for c in CAND}
    for p in panels:
        wds = [w for (pp, w) in groups if pp == p]; g = groups[(p, rng.choice(wds))]
        on = g[g.cond == "queue_onset"]; og = g[g.cond == "queue_ongoing"]
        if len(on) == 0 or len(og) == 0:
            continue
        io = rng.choice(len(on), min(5, len(on)), replace=False); ig = rng.choice(len(og), min(5, len(og)), replace=False)
        for c, (oc, gc) in CAND.items():
            per_panel[c].append(0.5 * (on[oc].to_numpy()[io].mean() + og[gc].to_numpy()[ig].mean()))
    for c in CAND:
        res[c].append(0.30 * np.mean(per_panel[c]))
S = {c: np.array(v) for c, v in res.items()}
print(f"E[S_main] (T2 part) {S['main'].mean():.4f}")
for c in [c for c in CAND if c != "main"]:
    d = S[c] - S["main"]; mx = np.maximum(S["main"], S[c]) - S["main"]
    print(f"{c}: E[hedge - main] {d.mean():+.4f} (sd {d.std():.4f}), E[max(main, hedge)] - E[main] {mx.mean():+.4f}, P(hedge > main) {np.mean(d > 0):.2f}")
