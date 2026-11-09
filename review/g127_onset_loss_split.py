"""G127 (10/30, ledger #81, strategy J21): split the onset loss of the precursor windows (M3 v3 DEV onset, J11's 88 windows
outside the 12 no-precursor ones; held-out days not read) into a site miss and a range error. Final A's onset flow (G120
path: q24f CF5 + physics onset adaptive a, evidence x0.25 -> best_set, k expectation, simulated blanks). Per window x k on
the T+30 row and the scored cells, 1 - IoU = (FP + FN) / union is cut by the true runs (runs of observed true cells, broken
by an observed free cell) and the selected runs (contiguous): site FN = FN of true runs the selection does not touch,
range FN = FN of touched true runs, site FP = FP of selected runs touching no true cell, range FP = FP of the others.
(i) site miss = site FN + site FP, (ii) range = range FN + range FP (sum = the loss). For each missed true run: rank r =
1 + the number of contiguous groups of unselected links with a higher probability than the run's best link (r <= 3 = a
near miss), known = every link of the run in the training-month onset vocabulary (rarity_table all-slot mean > 0),
J20 type = known and the window's 2-hour slot frequency 0 on the run.
usage: TFB_M3VER=v3 py g127_onset_loss_split.py"""
import os, sys, time, importlib.util
from collections import defaultdict
os.environ.setdefault("TFB_M3VER", "v3")
T00 = time.perf_counter(); R = r"C:/Users/user/tfb/"
import numpy as np, pandas as pd
spec = importlib.util.spec_from_file_location("g120", R + "review/g120_onset_gate.py"); g = importlib.util.module_from_spec(spec); spec.loader.exec_module(g)
G, m3v2, PO, best_set = g.G, g.m3v2, g.PO, g.best_set
J = pd.read_csv(g.W + "g117_onset_oracle.csv").set_index("w")
NOPRE = set(J[(J.v_ff >= 0.98) & J.flow_ratio.between(0.85, 1.15) & (J.fmax <= 0.1)].index)
RT = PO.rarity_table(); FREQ = {}
for (p, we, sl), f in RT.items():
    FREQ.setdefault(p, []).append(f)
FREQ = {p: np.mean(v, axis=0) for p, v in FREQ.items()}


def groups(b):
    """contiguous True runs of a bool vector -> list of index arrays"""
    ix = np.flatnonzero(b)
    return [] if not ix.size else np.split(ix, np.flatnonzero(np.diff(ix) > 1) + 1)


def true_runs(Y, e5):
    """observed true cells grouped into runs that only an observed free cell breaks"""
    ix = np.flatnonzero(Y); out, cur = [], [ix[0]] if ix.size else []
    for a, b in zip(ix[:-1], ix[1:]):
        if (e5[a + 1:b] & ~Y[a + 1:b]).any():
            out.append(np.array(cur)); cur = [b]
        else:
            cur.append(b)
    return out + ([np.array(cur)] if cur else [])


PN = g.nn_blend(); phys = g.lgb.Booster(model_file=g.W + "physon_base.txt"); ev = PO.month_events("validation", PO.rarity_table())
tick = time.perf_counter() - T00; rows, miss = [], []
for w in [w for w in g.DEV if w not in NOPRE]:
    m = G.META[w]; i = G.POS[w]; Xw = np.asarray(G.X[G.OFF[i]:G.OFF[i + 1]]); ph = phys.predict(PO.feats(Xw)); a = PO.adaptive_a(ev, m["day"])
    y5 = m["y"][:, 5].astype(bool); e5 = m["m"][:, 5].astype(bool); Y = y5 & e5; n = len(y5)
    P = m3v2._panel(m["panel"])[0]; key = (m["panel"], pd.Timestamp(P.times[m["t"]][:10]).dayofweek >= 5, (m["t"] % 288) // 24)
    fs = RT.get(key, np.zeros(n)); fa = FREQ[m["panel"]]; TR = true_runs(Y, e5)
    r = defaultdict(float, w=w, panel=m["panel"], day=m["day"], nruns=len(TR))
    for k, pr in m3v2.kdist(m["day"]).items():
        pb = PN[w].copy(); pb[5] = (1 - a) * pb[5] + a * ph; s, nd = m3v2.seen(m["panel"], m["t"], k, -6, 12)     # V.onset_pred "cur"
        if nd:
            pb[-1] = pb[-1] * np.where(G.dil(s, 2), 1.0, 0.25)
        sel = best_set(pb.ravel(), np.ones(pb.size, bool)).reshape(pb.shape); iou = G.iou(sel, m); s5 = sel[-1]; p5 = pb[-1]
        U = ((s5 | Y) & e5).sum(); FP = s5 & ~y5 & e5; parts = dict(sFN=0, rFN=0, sFP=0, rFP=0, sFP_adj=0)
        for R_ in TR:
            hit = s5[R_].any(); fn = int((~s5[R_]).sum()); parts["rFN" if hit else "sFN"] += fn
            if not hit:
                rank = 1 + len(groups((p5 > p5[R_].max()) & ~s5)); sp = np.arange(R_.min(), R_.max() + 1)
                known = bool((fa[sp] > 0).all()); j20 = known and bool((fs[sp] == 0).all())
                miss.append(dict(w=w, k=k, pr=pr, loss=pr * fn / U, rank=rank, near=rank <= 3, known=known, j20=j20, size=len(R_)))
        for S_ in groups(s5):
            fp = int(FP[S_].sum())
            if Y[S_].any():
                parts["rFP"] += fp
            else:
                parts["sFP"] += fp; parts["sFP_adj"] += int((FP[S_] & G.dil(Y, 2)[S_]).sum())
        tot = (parts["sFN"] + parts["rFN"] + parts["sFP"] + parts["rFP"]) / U
        assert abs(tot - (1 - iou)) < 1e-12, (w, k, tot, 1 - iou)
        r["iou"] += pr * iou
        for c in ("sFN", "rFN", "sFP", "rFP", "sFP_adj"):
            r[c] += pr * parts[c] / U
    rows.append(dict(r))
D = pd.DataFrame(rows); M = pd.DataFrame(miss)
ref = pd.read_csv(g.W + "g108_rectangles.csv").set_index("w")["A"]
print(f"{len(D)} precursor windows in {time.perf_counter() - T00:.0f}s (imports + NN blend {tick:.0f}s) | identity: IoU vs G108 column A max |d| "
      f"{float(np.max(np.abs(D.iou - ref.loc[D.w].to_numpy()))):.2e}; parts sum = 1 - IoU per window x k (asserted, 1e-12)")
D.to_csv(g.W + "g127_onset_loss_split.csv", index=False); M.to_csv(g.W + "g127_onset_missed_runs.csv", index=False)
loss = 1 - D.iou.mean(); si = (D.sFN + D.sFP).mean(); ra = (D.rFN + D.rFP).mean()
cl = (D.panel + "_" + D.day.astype(str)).to_numpy(); ids = pd.factorize(cl)[0]
Rm = np.random.default_rng(0).multinomial(ids.max() + 1, np.full(ids.max() + 1, 1 / (ids.max() + 1)), size=2000)


def bse(x):
    return float(((Rm @ np.bincount(ids, x)) / (Rm @ np.bincount(ids))).std())


print(f"loss {loss:.4f} (IoU {D.iou.mean():.4f}) = (i) site {si:.4f} ({si / loss:.0%}, SE {bse((D.sFN + D.sFP).to_numpy()):.4f}: missed true runs {D.sFN.mean():.4f}, "
      f"wrong-site picks {D.sFP.mean():.4f} of which within 2 links of the truth {D.sFP_adj.mean():.4f}) + (ii) range {ra:.4f} ({ra / loss:.0%}: FN {D.rFN.mean():.4f}, FP {D.rFP.mean():.4f})")
if len(M):
    tl = M.loss.sum() / len(D)
    print(f"missed true runs: {len(M.w.unique())} windows, site FN loss {tl:.4f} | share of that loss: near (r <= 3) {M.loss[M.near].sum() / M.loss.sum():.0%}, "
          f"known {M.loss[M.known].sum() / M.loss.sum():.0%}, known & near {M.loss[M.known & M.near].sum() / M.loss.sum():.0%}, J20 type {M.loss[M.j20].sum() / M.loss.sum():.0%}, "
          f"J20 type & near {M.loss[M.j20 & M.near].sum() / M.loss.sum():.0%}")
    print("rank distribution (loss-weighted):", M.groupby(M["rank"].clip(upper=8)).loss.sum().div(M.loss.sum()).round(2).to_dict())
print("windows by loss (top 12):"); print(D.assign(loss=1 - D.iou).sort_values("loss", ascending=False).head(12)[["w", "nruns", "iou", "sFN", "sFP", "sFP_adj", "rFN", "rFP"]].round(3).to_string(index=False))
