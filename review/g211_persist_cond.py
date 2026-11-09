"""G211 (11/02, ledger #177, ruling 745700): the selector keeps an ongoing window only if persistence (the queue at T - 5 held for six rows) has IoU <= 0.9 with the truth, so the forecast may be conditioned on that fact. From the probabilities best_set sees (A8's flow, B9s' flow on January, February), draw Y independently per cell (2,000 draws, seed 177), drop the draws with IoU(persistence, Y) > 0.9, take the marginals of the rest: pc = best_set on them; pm = the top k of them with k by the mean IoU over the kept draws. Below 50 kept draws the window stays as it is. G175's source with these columns; G175 and its csv are not changed. Dev windows only.
usage: TFB_X3=x3af TFB_M3VER=v3 py -B -X utf8 g211_persist_cond.py"""
import sys
s = open(r"C:/Users/user/tfb/review/g175_bundle_read.py", encoding="utf-8").read()


def sub(a, b, n=1):
    global s
    assert s.count(a) == n, (a[:60], s.count(a))
    s = s.replace(a, b)


H = '''RNG = np.random.default_rng(177); REJ = []


def cond_p(pb, s0, thr=0.9, N=2000):
    Y = RNG.random((N,) + pb.shape) < pb; S = s0[None, None, :]; I = (Y & S).sum(axis=(1, 2)); U = (Y | S).sum(axis=(1, 2)); keep = I <= thr * U; REJ.append(1 - float(keep.mean()))
    if keep.sum() < 50 or keep.all():
        return pb, best_set(pb.ravel(), np.ones(pb.size, bool)).reshape(pb.shape)
    Yk = Y[keep].reshape(int(keep.sum()), -1); pc = Yk.mean(axis=0); order = np.argsort(-pc)[:600]; Ik = np.cumsum(Yk[:, order], axis=1); ny = Yk.sum(axis=1, keepdims=True); kk = np.arange(1, len(order) + 1)[None, :]; v = (Ik / np.maximum(kk + ny - Ik, 1)).mean(axis=0); sel = np.zeros(pb.size, bool); sel[order[:int(np.argmax(v)) + 1]] = True
    return pc.reshape(pb.shape), sel.reshape(pb.shape)


def frame(wins, parts):'''
sub('def frame(wins, parts):', H)
sub('CC = ("A", "bundle", "B", "Bc", "B2", "Bc2"); cols', 'CC = ("A", "bundle", "B", "Bc", "B2", "Bc2", "pc", "pm"); cols')
sub('sc["bundle"] += pr * G.iou(dec(tmp((0.8 * ((1 - X3) * (0.8 * xs + 0.2 * pt) + X3 * px) + 0.2 * pl) * cf * sw)), m)', 'sc["bundle"] += pr * G.iou(dec(tmp((0.8 * ((1 - X3) * (0.8 * xs + 0.2 * pt) + X3 * px) + 0.2 * pl) * cf * sw)), m); pc_, ms_ = cond_p(tmp((0.8 * ((1 - X3) * (0.8 * xs + 0.2 * pt) + X3 * px) + 0.2 * pl) * cf * sw), q[t - 1]); sc["pc"] += pr * G.iou(dec(pc_), m); sc["pm"] += pr * G.iou(g132.gate(ms_, ex, nr), m)\n            ')
sub('r["B"] += pr * h.iou(h.decode(p_), y, e, "queue_ongoing");', 'pc_, ms_ = cond_p(tmp((1 - X3) * p_ + X3 * px), c_.q[t - 1]); r["pc"] = r.get("pc", 0.0) + pr * h.iou(h.decode(pc_), y, e, "queue_ongoing"); r["pm"] = r.get("pm", 0.0) + pr * h.iou(ms_, y, e, "queue_ongoing")\n                    r["B"] += pr * h.iou(h.decode(p_), y, e, "queue_ongoing");')
sub('"g175_dev.csv"', '"g211_dev.csv"', 2); sub('"g175_train_B.csv"', '"g211_train.csv"')
sub('for a, b, nm in (("A", "bundle", "A side"), ("B", "Bc", "B side"), ("B2", "Bc2", "B side with X1 beta 0.25")):\n        s = m3v2.summary(D, a, b, "private"); sm =', 'r_ = np.array(REJ); print(f"rejected share: median {np.median(r_):.3f}, mean {r_.mean():.3f}, share of decodes > 0.05 {(r_ > 0.05).mean():.3f}, > 0.5 {(r_ > 0.5).mean():.3f}, all kept {(r_ == 0).mean():.3f}", flush=True)\n    for a, b, nm in (("bundle", "pc", "A8 marginals"), ("bundle", "pm", "A8 expected IoU")):\n        s = m3v2.summary(D, a, b, "private"); sm =')
sub('print(mo, len(d), f"B raw {d.B.mean():.4f} Bc - B {(d.Bc - d.B).mean():+.4f} (SE {boot((d.Bc - d.B).to_numpy(), d.panel + \'_\' + d.day.astype(str)):.4f})", flush=True)', 'print(mo, len(d), " | ".join(f"{c_} {(d[c_] - d.Bc).mean():+.4f} (SE {boot((d[c_] - d.Bc).to_numpy(), d.panel + \'_\' + d.day.astype(str)):.4f})" for c_ in ("pc", "pm")), flush=True)')
sub('z = s["t2_weighted"] / s["se"]', 'z = s["t2_weighted"] / max(s["se"], 1e-9)')
sys.argv = [sys.argv[0], "dev"]; exec(compile(s, "g175p", "exec"))
