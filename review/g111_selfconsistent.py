"""G111 (10/29, strategy upper-bound audit item 4): the self-consistent bound of Task 2 on M3 v3 dev windows. Under the final
A flow (G93 / G108 harness: onset = q24f CF5 + physics onset, evidence rule, decoder; ongoing = X1 + LightGBM 0.2 + cone,
evidence weight, decoder, far gate) take the probabilities the decoder sees and the decoded set S. If the probabilities
were the truth, the expected IoU of S is E_p[IoU(S, Y)]; compare with the realised IoU(S, y) (scored cells only, k
expectation). Y is drawn two ways that bracket the dependence between cells: independent Bernoulli per cell, and one
uniform per window shared by all cells (comonotone; jams are whole-run events, B13b). 200 draws each.
Equal -> the decoder is consistent with the model and the rest of the loss is information; realised < expected ->
the probabilities are over-confident where it matters (calibration-in-the-large is 0.96 / 1.01, #54).
usage: TFB_M3VER=v3 py g111_selfconsistent.py"""
import os, sys
os.environ.setdefault("TFB_M3VER", "v3")
sys.argv = [sys.argv[0]]
src = open(r"C:/Users/user/tfb/review/g108_rectangles.py", encoding="utf-8").read().split("\ncols = {c: {} for c")[0]
exec(compile(src, "g108", "exec"))
rng = np.random.default_rng(111); ND = 200


def iou_masked(sel, Y, mk):
    """sel (r, n) bool; Y (d, r, n) bool draws; mk (r, n) scored -> (d,) IoU"""
    s = sel & mk; y = Y & mk[None]
    u = (s[None] | y).sum((1, 2)); i = (s[None] & y).sum((1, 2))
    return np.where(u == 0, 1.0, i / np.maximum(u, 1))


acc = {}
for i, w in enumerate(G.WID):
    m = G.META[w]
    if m3v2.is_holdout(np.array([m["day"]]))[0]:
        continue
    Xw = np.asarray(G.X[G.OFF[i]:G.OFF[i + 1]]); og = m["cond"] == "queue_ongoing"
    y, mk = m["y"].T.astype(bool), m["m"].T.astype(bool); kd = m3v2.kdist(m["day"])
    if not og:
        pn = sum(wt * PRO[t][w] for t, wt in ON) / sum(wt for _, wt in ON); ph = phys.predict(PO.feats(Xw)); a_ph = PO.adaptive_a(ev, m["day"])
        pl = None
    else:
        pl = FS.lgb_ongoing(Xw)
    real = ei = ec = 0.0
    for k, pr in kd.items():
        if og:
            p = (0.8 * X1[(w, k)] + 0.2 * pl) * cone_f(m, k)
            s_, nd = m3v2.seen(m["panel"], m["t"], k, -18, 12)
            pb = p * np.where(G.dil(s_, 1), 1.0, 0.85)[None, :] if nd else p.copy()
            sel = V.far_gate(best_set(pb.ravel(), np.ones(pb.size, bool)).reshape(pb.shape), m, k)
            sl, yy, mm, pp = sel, y, mk, np.clip(pb, 0, 1)
        else:
            p = pn.copy(); p[5] = (1 - a_ph) * p[5] + a_ph * ph
            pb = p.copy(); s_, nd = m3v2.seen(m["panel"], m["t"], k, -6, 12)
            if nd:
                pb[-1] = pb[-1] * np.where(G.dil(s_, 2), 1.0, 0.25)
            sel = best_set(pb.ravel(), np.ones(pb.size, bool)).reshape(pb.shape)
            sl, yy, mm, pp = sel[-1:], y[-1:], mk[-1:], np.clip(pb[-1:], 0, 1)
        real += pr * G.iou(sel, m)
        U = rng.random((ND,) + pp.shape); ei += pr * iou_masked(sl, U < pp[None], mm).mean()
        u1 = rng.random((ND, 1, 1)); ec += pr * iou_masked(sl, u1 < pp[None], mm).mean()
    a = acc.setdefault(m["cond"], []); a.append((real, ei, ec))
for c, v in acc.items():
    v = np.array(v); d1, d2 = v[:, 1] - v[:, 0], v[:, 2] - v[:, 0]
    print(f"{c}: dev windows {len(v)} | realised {v[:, 0].mean():.4f} | expected under the model: independent {v[:, 1].mean():.4f} "
          f"(gap {d1.mean():+.4f} SE {d1.std(ddof=1) / np.sqrt(len(v)):.4f}), comonotone {v[:, 2].mean():.4f} (gap {d2.mean():+.4f} SE {d2.std(ddof=1) / np.sqrt(len(v)):.4f})")
