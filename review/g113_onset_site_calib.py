"""G113 (10/29, ledger #58): site-level calibration of the onset probabilities the decoder sees. best_set picks the number
of cells from the absolute probabilities, and at the site level they are under-confident after the evidence rule (G112:
top-site p 0.23 holds the queue 0.70). Per M3 v3 dev onset window and k: p_s = max of the T+30 probabilities (after the
evidence rule) over each known site's rectangle (b13_sites), label = a queued label cell in the rectangle. Isotonic
c(p) fitted on the dev windows of the other day parity (cross-fitting); cells of each rectangle x c(p_s) / p_s (capped
at 1); other cells unchanged; the same decoder. Column A = the final A onset flow, S = calibrated.
Writes work/t2/g113_onset_site_calib.csv for judge.py --cond queue_onset.
usage: TFB_M3VER=v3 py g113_onset_site_calib.py"""
import os, sys
os.environ.setdefault("TFB_M3VER", "v3")
sys.argv = [sys.argv[0]]
src = open(r"C:/Users/user/tfb/review/g108_rectangles.py", encoding="utf-8").read().split("\ncols = {c: {} for c")[0]
exec(compile(src, "g108", "exec"))
from sklearn.isotonic import IsotonicRegression

WIN = []
for i, w in enumerate(G.WID):
    m = G.META[w]
    if m["cond"] != "queue_onset" or m3v2.is_holdout(np.array([m["day"]]))[0]:
        continue
    Xw = np.asarray(G.X[G.OFF[i]:G.OFF[i + 1]])
    pn = sum(wt * PRO[t][w] for t, wt in ON) / sum(wt for _, wt in ON); ph = phys.predict(PO.feats(Xw)); a_ph = PO.adaptive_a(ev, m["day"])
    sites = sorted((h, tl) for (pp, h), tl in SITE.items() if pp == m["panel"]); y = m["y"][:, 5].astype(bool)
    per_k = {}
    for k, pr in m3v2.kdist(m["day"]).items():
        p = pn.copy(); p[5] = (1 - a_ph) * p[5] + a_ph * ph
        s_, nd = m3v2.seen(m["panel"], m["t"], k, -6, 12)
        if nd:
            p[-1] = p[-1] * np.where(G.dil(s_, 2), 1.0, 0.25)
        per_k[k] = (pr, p)
    WIN.append((w, m, sites, y, per_k))
cols = {"A": {}, "S": {}}
for par in (0, 1):
    fit = [(per_k[k][1][-1][tl:h + 1].max(), float(y[tl:h + 1].any()), per_k[k][0])
           for w, m, sites, y, per_k in WIN if m["day"] % 2 != par for k in per_k for h, tl in sites]
    f = np.array(fit); iso = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip").fit(f[:, 0], f[:, 1], sample_weight=f[:, 2])
    for w, m, sites, y, per_k in WIN:
        if m["day"] % 2 != par:
            continue
        sa = ss = 0.0
        for k, (pr, p) in per_k.items():
            sa += pr * G.iou(best_set(p.ravel(), np.ones(p.size, bool)).reshape(p.shape), m)
            q = p.copy()
            for h, tl in sites:
                ps = p[-1][tl:h + 1].max()
                if ps > 1e-6:
                    q[-1][tl:h + 1] = np.minimum(1.0, p[-1][tl:h + 1] * iso.predict([ps])[0] / ps)
            ss += pr * G.iou(best_set(q.ravel(), np.ones(q.size, bool)).reshape(q.shape), m)
        cols["A"][w] = sa; cols["S"][w] = ss
dmax = max(abs(cols["A"][w] - REF.loc[w, "phys"]) for w in cols["A"])
print(f"identity: A vs G93 phys: max |d| {dmax:.2e}")
D = V.frame(cols); D = D[D.cond == "queue_onset"]; D.to_csv(W + "g113_onset_site_calib.csv", index=False)
dv = ~m3v2.is_holdout(D.day); print(f"dev onset {int(dv.sum())}: A {D[dv].A.mean():.4f} | site-calibrated {D[dv].S.mean():.4f}")
