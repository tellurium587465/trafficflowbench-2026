"""E3 on M3 (pre-registered STATE.md 10/22 11:00): same-month prior correction from observed days.
For each M3 window, over the earlier-day pseudo-origins at the same clock time (t2_m3_prev_build.py, k-day limit and
trap 4 applied), per link x horizon step:
  k_q = observed-and-queued days, k_o = observed days, pbar = the model's own mean prediction on the observed days
  pi  = (k_q + a pbar) / (k_o + a)          (beta-binomial posterior mean, a = prior strength)
  logit p' = logit p + beta (logit pi - logit pbar) + c     where k_o > 0, unchanged elsewhere
Variants: REPL (E3 replaces the evidence rule: onset -> decode; ongoing -> decode + far gate) and ADD (E3, then the
evidence rule x0.25 / x0.85, then decode / gate). Baseline = the current rule. (a, beta, c) chosen on the other panels
(leave-one-panel-out, grid a {0.5,1,2,4} x beta {0,.25,.5,.75,1} x c {-0.5,0,0.5}, per condition), scored on the held-out
panel. Model q24f, seeds 0-3, both conditions. Cluster bootstrap (panel x day) for the paired differences.
usage: python t2_m3_e3.py"""
import sys, pickle, itertools
sys.argv = [sys.argv[0], "0,1,2,3"]
import numpy as np, pandas as pd
import t2_m3_gate as G

W = G.W
PM = pickle.load(open(W + "m3prev_meta.pkl", "rb"))
parts = [(w, pickle.load(open(W + f"probs_m3prev__{f}_s{s}.pkl", "rb"))) for s in G.SEEDS for f, w in G.CFG5("q24f")]
tw = sum(w for w, _ in parts); PP = {k: sum(w * pr[k] for w, pr in parts) / tw for k in parts[0][1]}
STAT = {}
for e in PM:                                              # stats per parent window: k_q, k_o, sum of pbar on observed days
    y, m = e["y"].T.astype(float), e["m"].T.astype(float); p = PP[e["window_id"]]
    kq, ko, sp = STAT.get(e["parent"], (0.0, 0.0, 0.0))
    STAT[e["parent"]] = (kq + y * m, ko + m, sp + p * m)
lg = lambda x: np.log(x / (1 - x))
clip = lambda x: np.clip(x, 1e-4, 1 - 1e-4)


def corrected(p_, w, a, beta, c):
    if w not in STAT or beta == 0 and c == 0:
        return p_
    kq, ko, sp = STAT[w]; has = ko > 0
    pbar = clip(np.where(has, sp / np.maximum(ko, 1), 0.5)); pi = clip((kq + a * pbar) / (ko + a))
    z = lg(clip(p_)) + np.where(has, beta * (lg(pi) - lg(pbar)) + c, 0.0)
    out = 1 / (1 + np.exp(-z))
    return np.where(p_ > 0, out, 0.0)                     # keep the onset rows that the decoder zeroes (p[:5] = 0)


def predict(w, m, a, beta, c, variant):
    p_ = corrected(G.NOMEM[w], w, a, beta, c)
    if m["cond"] == "queue_onset":
        return G.onset_pred(p_, m, "cur" if variant == "ADD" else "none")
    if variant == "ADD":
        return G.ongoing_pred(p_, m, True)
    sel = G.best_set(p_.ravel(), np.ones(p_.size, bool)).reshape(p_.shape)
    return G.far_gate(sel, m)


if __name__ == "__main__":
    GRID = list(itertools.product([0.5, 1, 2, 4], [0, 0.25, 0.5, 0.75, 1], [-0.5, 0, 0.5]))
    rows = []
    for cond in ("queue_onset", "queue_ongoing"):
        ws = [w for w in G.WID if G.META[w]["cond"] == cond]
        base = {w: G.iou(G.onset_pred(G.NOMEM[w], G.META[w], "cur") if cond == "queue_onset" else G.ongoing_pred(G.NOMEM[w], G.META[w], True), G.META[w]) for w in ws}
        for variant in ("REPL", "ADD"):
            score = {g: {w: G.iou(predict(w, G.META[w], *g, variant), G.META[w]) for w in ws} for g in GRID}
            pans = sorted({G.META[w]["panel"] for w in ws})
            chosen = {}
            for ph in pans:                                   # leave-one-panel-out choice of (a, beta, c)
                tr = [w for w in ws if G.META[w]["panel"] != ph]
                chosen[ph] = max(GRID, key=lambda g: np.mean([score[g][w] for w in tr]))
            for w in ws:
                g = chosen[G.META[w]["panel"]]; m = G.META[w]
                rows.append(dict(cond=cond, variant=variant, cl=f"{m['panel']}_{m['day']}", band="0.5-0.6" if m["cov"] < 0.6 else "0.6-0.7",
                                 base=base[w], e3=score[g][w], k=m["k"], a=g[0], beta=g[1], c=g[2]))
            print(cond, variant, "choices per held-out panel", {p: chosen[p] for p in pans}, flush=True)
    D = pd.DataFrame(rows); D.to_csv(W + "m3_e3.csv", index=False); rng = np.random.default_rng(0)
    for (cond, variant), g in D.groupby(["cond", "variant"]):
        u = g.cl.unique(); idx = {x: np.flatnonzero(g.cl.to_numpy() == x) for x in u}; d = (g.e3 - g.base).to_numpy()
        se = np.std([d[np.concatenate([idx[x] for x in rng.choice(u, len(u))])].mean() for _ in range(1000)])
        print(f"{cond} {variant}: windows {len(g)}, base {g.base.mean():.4f}, E3 (nested) {g.e3.mean():.4f}, diff {d.mean():+.4f} "
              f"(SE {se:.4f}, {d.mean() / max(se, 1e-9):+.1f} SE); windows with k>0 {int((g.k > 0).sum())}", flush=True)
    for (cond, variant, band), g in D.groupby(["cond", "variant", "band"]):
        print(f"  {cond} {variant} band {band}: windows {len(g)}, diff {(g.e3 - g.base).mean():+.4f}")
