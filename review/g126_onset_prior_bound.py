"""G126 (10/30, ledger #80, strategy J20 (a)): is J11's "no precursor" onset share really free of <= T information?
On J11's 12 no-precursor M3 v3 DEV onset windows (speed >= 0.98 vff, upstream flow 0.85-1.15, site frequency <= 0.1;
held-out days not read) take the true T+30 span [lo, hi] (length L, assumed known) and slide it over the window under
three location priors: U = uniform over the window's positions, V = uniform over the positions whose every link is in the
training-month onset vocabulary (rarity_table, mean frequency over all slots > 0), F = proportional to that mean frequency.
For each prior the contiguous set with the highest expected IoU (exhaustive) gives the bound of a method without location
information (lenient: it knows L), and that set is also scored on the actual label. IoU on all cells (the realised IoU of
final A is on the scored cells: approximation, noted). Diagnostic: the k weight with which the same-month evidence of the
earlier days (m3v2.seen -6..+12, dilated by 2 links) covers the label, and the realised IoU by that.
usage: TFB_M3VER=v3 py g126_onset_prior_bound.py"""
import os, sys, time
os.environ.setdefault("TFB_M3VER", "v3")
sys.path.insert(0, r"C:/Users/user/tfb/src"); argv0 = sys.argv[0]; sys.argv = [argv0, "0"]
T00 = time.perf_counter()
import numpy as np, pandas as pd
import t2_m3_gate as G, m3v2, t2_phys_onset as PO

A = pd.read_csv(G.W + "g108_rectangles.csv").set_index("w")["A"]
J = pd.read_csv(G.W + "g117_onset_oracle.csv").set_index("w"); J = J[J.nY > 0]
print(f"identity: g117 iou vs g108 column A max |d| {float(np.max(np.abs(J.iou - A.loc[J.index]))):.2e}")
U12 = J[(J.v_ff >= 0.98) & J.flow_ratio.between(0.85, 1.15) & (J.fmax <= 0.1)].index.tolist()
RT = PO.rarity_table(); FREQ = {}
for (p, we, sl), f in RT.items():
    FREQ.setdefault(p, []).append(f)
FREQ = {p: np.mean(v, axis=0) for p, v in FREQ.items()}


def best(n, L, wa):
    """wa (n - L + 1,) prior over the truth start a -> (best expected IoU, (b, L')) over contiguous sets [b, b + L')"""
    a = np.arange(n - L + 1); b = np.arange(n)[:, None, None]; lp = np.arange(1, n + 1)[None, :, None]; aa = a[None, None, :]
    inter = np.clip(np.minimum(aa + L, b + lp) - np.maximum(aa, b), 0, None); e = (inter / (L + lp - inter)) @ wa
    e = np.where(b[:, :, 0] + lp[:, :, 0] <= n, e, -1.0); i, j = np.unravel_index(np.argmax(e), e.shape)
    return float(e[i, j]), (int(i), int(j) + 1)


def iou_cells(s, y, mk=None):
    if mk is not None:
        s, y = s & mk, y & mk
    u = (s | y).sum(); return 1.0 if u == 0 else float((s & y).sum() / u)


rng = np.random.default_rng(126); rows = []
for w in U12:
    m = G.META[w]; n = m["y"].shape[0]; y5 = m["y"][:, 5].astype(bool); e5 = m["m"][:, 5].astype(bool); Y = np.flatnonzero(y5 & e5)
    lo, hi = Y.min(), Y.max(); L = hi - lo + 1; ys = np.zeros(n, bool); ys[lo:hi + 1] = True
    fr = FREQ.get(m["panel"], np.zeros(n))[:n]; voc = fr > 0; a = np.arange(n - L + 1)
    allv = np.array([voc[x:x + L].all() for x in a]); mf = np.array([fr[x:x + L].mean() for x in a])
    pri = dict(U=np.ones(len(a)), V=allv.astype(float) if allv.any() else np.array([voc[x:x + L].any() for x in a], float),
               F=mf if mf.sum() > 0 else np.ones(len(a)))
    r = dict(w=w, panel=m["panel"], day=m["day"], n=n, L=L, nY=len(Y), real=float(A[w]), vocab=int(voc.sum()),
             truth_in_vocab=bool(voc[lo:hi + 1].all()), truth_freq=float(fr[lo:hi + 1].mean()))
    for k_, wa in pri.items():
        wa = wa / wa.sum(); e, (b, lp) = best(n, L, wa); s = np.zeros(n, bool); s[b:b + lp] = True
        r[f"E_{k_}"] = e; r[f"hit_{k_}"] = iou_cells(s, ys); r[f"hitm_{k_}"] = iou_cells(s, y5, e5)
        if k_ == "U":                                                  # Monte Carlo check of the closed form
            st = rng.choice(a, size=10 ** 4, p=wa); inter = np.clip(np.minimum(st + L, b + lp) - np.maximum(st, b), 0, None)
            r["E_U_mc"] = float(np.mean(inter / (L + lp - inter)))
    cov = 0.0
    for k, pr in m3v2.kdist(m["day"]).items():
        sn, nd = m3v2.seen(m["panel"], m["t"], k, -6, 12)
        cov += pr * float(nd > 0 and G.dil(sn, 2)[Y].all())
    r["ev_cover"] = cov; rows.append(r)
D = pd.DataFrame(rows); D["E_max"] = D[["E_U", "E_V", "E_F"]].max(axis=1); D["d"] = D.real - D.E_max
D.to_csv(G.W + "g126_onset_prior_bound.csv", index=False)
print(f"{len(D)} windows in {time.perf_counter() - T00:.0f}s | E_U closed form vs 10^4 draws max |d| {float(np.max(np.abs(D.E_U - D.E_U_mc))):.4f}")
pd.set_option("display.width", 250)
print(D[["w", "n", "L", "vocab", "truth_in_vocab", "truth_freq", "real", "E_U", "E_V", "E_F", "hit_U", "hit_V", "hit_F", "hitm_F", "ev_cover"]].round(3).to_string(index=False))
bs = np.random.default_rng(0).choice(len(D), size=(4000, len(D))); se = float(D.d.to_numpy()[bs].mean(1).std())
print(f"realised {D.real.mean():.3f} | prior bound E_U {D.E_U.mean():.3f} E_V {D.E_V.mean():.3f} E_F {D.E_F.mean():.3f} E_max {D.E_max.mean():.3f} | "
      f"prior-only on the label: U {D.hit_U.mean():.3f} V {D.hit_V.mean():.3f} F {D.hit_F.mean():.3f} (scored cells F {D.hitm_F.mean():.3f})")
print(f"d = realised - E_max {D.d.mean():+.3f} (SE {se:.3f}, z {D.d.mean() / se:+.1f})")
c = D.ev_cover >= 0.5
print(f"evidence covers the label (k weight >= 0.5): {int(c.sum())}/{len(D)} windows, realised {D.real[c].mean() if c.any() else float('nan'):.3f} | "
      f"not covered: {int((~c).sum())}, realised {D.real[~c].mean() if (~c).any() else float('nan'):.3f}")
