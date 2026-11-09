"""G112 (10/29, strategy upper-bound audit, onset at the site level): with the generator's onset structure given (a queue at
T+30 in every onset window, made of site rectangles, birth extent a site constant: #49 / #51), the onset uncertainty is
which sites are born. Per M3 v3 dev onset window and k, from the final A onset probabilities at T+30 (q24f CF5 + physics
onset, after the evidence rule): each known site s (b13_sites, training months) in the panel is born with
p_s = max of the probabilities over its rectangle, independently; links outside every known rectangle stay independent
cells with their own probability; draws with no queued cell are rejected (the window guarantees one). Expected IoU of the
decoded set = mean over 400 draws of IoU(decoded, union of the born rectangles + cell draws) on the scored cells; compare
with the realised IoU. Also the calibration of the top site: p of the most likely site vs the share of windows whose
label holds that site's head.
usage: TFB_M3VER=v3 py g112_onset_site_bound.py"""
import os, sys
os.environ.setdefault("TFB_M3VER", "v3")
sys.argv = [sys.argv[0]]
src = open(r"C:/Users/user/tfb/review/g108_rectangles.py", encoding="utf-8").read().split("\ncols = {c: {} for c")[0]
exec(compile(src, "g108", "exec"))
rng = np.random.default_rng(112); ND = 400
acc, top, top0 = [], [], []
for i, w in enumerate(G.WID):
    m = G.META[w]
    if m["cond"] != "queue_onset" or m3v2.is_holdout(np.array([m["day"]]))[0]:
        continue
    Xw = np.asarray(G.X[G.OFF[i]:G.OFF[i + 1]]); y, mk = m["y"][:, 5].astype(bool), m["m"][:, 5].astype(bool); n = len(y)
    pn = sum(wt * PRO[t][w] for t, wt in ON) / sum(wt for _, wt in ON); ph = phys.predict(PO.feats(Xw)); a_ph = PO.adaptive_a(ev, m["day"])
    sites = sorted((h, tl) for (pp, h), tl in SITE.items() if pp == m["panel"])
    real = ex = 0.0
    for k, pr in m3v2.kdist(m["day"]).items():
        p = pn.copy(); p[5] = (1 - a_ph) * p[5] + a_ph * ph
        pb = p.copy(); s_, nd = m3v2.seen(m["panel"], m["t"], k, -6, 12)
        if nd:
            pb[-1] = pb[-1] * np.where(G.dil(s_, 2), 1.0, 0.25)
        sel = best_set(pb.ravel(), np.ones(pb.size, bool)).reshape(pb.shape)
        real += pr * G.iou(sel, m); q = np.clip(pb[-1], 0, 1); a = sel[-1] & mk
        insite = np.zeros(n, bool); ps = []
        for h, tl in sites:
            insite[tl:h + 1] = True; ps.append(q[tl:h + 1].max())
        ps = np.array(ps); vals = []
        while len(vals) < ND:
            Y = rng.random(n) < np.where(insite, 0.0, q)
            born = rng.random(len(sites)) < ps
            for (h, tl), b in zip(sites, born):
                if b:
                    Y[tl:h + 1] = True
            if not Y.any():
                continue
            yy = Y & mk; u = (a | yy).sum(); vals.append(1.0 if u == 0 else (a & yy).sum() / u)
        ex += pr * float(np.mean(vals))
        if k == max(m3v2.kdist(m["day"]), key=lambda kk: m3v2.kdist(m["day"])[kk]) and len(sites):
            j = int(np.argmax(ps)); h, tl = sites[j]; top.append((ps[j], bool(y[tl:h + 1].any())))
            q0 = np.clip(p[-1], 0, 1); ps0 = np.array([q0[tl_:h_ + 1].max() for h_, tl_ in sites]); j0 = int(np.argmax(ps0))
            h0, tl0 = sites[j0]; top0.append((ps0[j0], bool(y[tl0:h0 + 1].any())))   # before the evidence rule
    acc.append((real, ex))
v = np.array(acc); d = v[:, 1] - v[:, 0]
print(f"dev onset windows {len(v)} | realised {v[:, 0].mean():.4f} | expected under site-level independence (>= 1 queue) {v[:, 1].mean():.4f} | "
      f"gap {d.mean():+.4f} (SE {d.std(ddof=1) / np.sqrt(len(v)):.4f})")
t = np.array(top, float)
for lo, hi in ((0, .5), (.5, .8), (.8, .95), (.95, 1.01)):
    s = (t[:, 0] >= lo) & (t[:, 0] < hi)
    if s.any():
        print(f"  top site p in [{lo:.2f}, {hi:.2f}): windows {int(s.sum())} | mean p {t[s, 0].mean():.3f} | label holds the site {t[s, 1].mean():.3f}")
print("before the evidence rule (network + physics mix only):")
t = np.array(top0, float)
for lo, hi in ((0, .5), (.5, .8), (.8, .95), (.95, 1.01)):
    s_ = (t[:, 0] >= lo) & (t[:, 0] < hi)
    if s_.any():
        print(f"  top site p in [{lo:.2f}, {hi:.2f}): windows {int(s_.sum())} | mean p {t[s_, 0].mean():.3f} | label holds the site {t[s_, 1].mean():.3f}")
