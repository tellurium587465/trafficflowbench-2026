"""G269 (11/04, ledger #227, diagnostic first, declared by the strategy before the results, auditor L1): what the far gate drops in the
band 4-8 links upstream of the queue's tail. Tails = the upstream ends of the last-queue runs (lq: each link's latest finite state at
T-60 .. T-5; links ordered upstream -> downstream by order_index). For each dropped cell (selected by best_set, removed by the far gate):
its distance upstream of the nearest tail downstream of it; per band the number of dropped cells, their truth rate, their mean
probability, and the decode's threshold (the window's smallest selected probability before the gate).
  jf : the 316 January / February selector ongoing windows, G229's flow (capB0 s0-3, G119's SCG flow, the row T as the last queue),
       the k draw with the largest weight
  dev: M3 v3 dev 356 windows, A12's flow of G212 (A13 on dev), every k of the draw weighted
usage: TFB_X3=x3af TFB_M3VER=v3 py -B -X utf8 g269_far_gate_band.py jf | dev"""
import os, sys, pickle, importlib.util
os.environ.setdefault("TFB_M3VER", "v3")
import numpy as np, pandas as pd
R = r"C:/Users/user/tfb/"; MODE = sys.argv[1]


def tails(lq):
    return [j for j in range(len(lq)) if lq[j] and (j == 0 or not lq[j - 1])]


def band_stats(drop, y, pb, lq, thr, acc, w=1.0):
    tl = tails(lq)
    for r, j in zip(*np.nonzero(drop)):
        dn = [t_ - j for t_ in tl if t_ > j]; d = min(dn) if dn else -1
        b = "4-8 up" if 4 <= d <= 8 else ("1-3 up" if 1 <= d <= 3 else (">8 up" if d > 8 else "other"))
        a = acc.setdefault(b, [0.0, 0.0, 0.0, 0.0]); a[0] += w; a[1] += w * float(y[r, j]); a[2] += w * float(pb[r, j]); a[3] += w * thr


def report(acc, lab):
    print(f"--- {lab}: far-gate drops by band (weighted cells | truth rate | mean p | decode threshold)", flush=True)
    for b in ("1-3 up", "4-8 up", ">8 up", "other"):
        if b in acc and acc[b][0] > 0:
            n, t_, p_, th = acc[b]; print(f"  {b:7s} {n:8.1f} | {t_ / n:.3f} | {p_ / n:.3f} | {th / n:.3f}", flush=True)


if MODE == "jf":
    a_ = sys.argv; sys.argv = ["g119", "run"]; spec = importlib.util.spec_from_file_location("g119", R + "review/g119_month_postproc.py"); g = importlib.util.module_from_spec(spec); spec.loader.exec_module(g); sys.argv = a_
    T, MON = g.T, g.MON; PR = [pickle.load(open(g.W + f"probs_jfmh0__capB0_s{s}_fold2030-12.pkl", "rb")) for s in range(4)]; dec0 = g.decode; LAST = {}
    g.decode = lambda pb: (LAST.__setitem__("pb", pb), dec0(pb))[1]; acc = {}
    KS = [k for k in np.flatnonzero(T.META["is_sel"] & np.isin(MON, g.MONTHS)) if str(T.META["cond"][k]) == "queue_ongoing"]
    for mo in g.MONTHS:
        for p in sorted({str(T.META["panel"][k]) for k in KS if MON[k] == mo}):
            P = g.load(p, "train"); key = (mo, p, "mask"); g.CTX.clear(); g.sim_blank.cache_clear(); g.CTX[key] = g.Ctx(key, P, np.where(P.blank, np.nan, P.V), P.V); c = g.CTX[key]
            for k in [k for k in KS if MON[k] == mo and str(T.META["panel"][k]) == p]:
                t = int(T.META["t"][k]); dom = int(P.times[t][8:10]) - 1; lab, el = T.LAB[k]; y = lab.astype(bool) & el.astype(bool)
                lq = np.asarray(g.XN[g.OFFN[k]:g.OFFN[k + 1]][:, g.G.CI], float) == 1; ot = c.obs[t].astype(bool); lq2 = np.where(ot, c.q[t].astype(bool), lq)
                p_ = sum(d[str(k)] for d in PR) / len(PR); cone = g.cone_parts(c, t, lq2, P.length); kd = g.m3v2.kdist(dom, cap=3); kk = max(kd, key=kd.get)
                pin = p_ * g.cone_f(c, cone, t, kk); s0 = g.ongoing_pred(c, pin, t, kk, False, lq2); pb = LAST["pb"]; s1 = g.ongoing_pred(c, pin, t, kk, True, lq2)
                thr = float(pb[s0].min()) if s0.any() else 1.0; band_stats(s0 & ~s1, y, pb, lq2, thr, acc)
            g.load.cache_clear()
    report(acc, "January / February 316 windows")
    sys.exit()
assert MODE == "dev"
R_ = R + "review/"; END = 'exec(compile(s, "g175a", "exec"))'; src = open(R_ + "g212_a10_read.py", encoding="utf-8").read(); assert src.count(END) == 1
exec(compile(src.replace(END, ""), "g212", "exec"))
FA = '(0.8 * (0.8 * ((1 - X3) * (0.8 * xs + 0.2 * pt) + X3 * px) + 0.2 * pu) + 0.2 * pl) * cf * sw'
sub('sc["a12"] += pr * G.iou(dec(tmp(' + FA + ')), m)',
    'sc["a12"] += pr * G.iou(dec(tmp(' + FA + ')), m); pb_ = tmp(' + FA + '); s0_ = best_set(pb_.ravel(), np.ones(pb_.size, bool)).reshape(pb_.shape); s1_ = dec(pb_); '
    'yy_ = m["y"].T.astype(bool) & m["m"].T.astype(bool); band_stats(s0_ & ~s1_, yy_, pb_, lq, float(pb_[s0_].min()) if s0_.any() else 1.0, ACC, pr)')
sub('D.to_csv(W + "g212_dev.csv"', 'D.to_csv(W + "g269_dev.csv"')
sub('    Df = train_B(); Df.to_csv(W + "g212_train.csv", index=False)\n', '    report(ACC, "M3 dev 356 windows (k-weighted)")\n    sys.exit()\n')
ACC = {}
exec(compile(s, "g175f", "exec"))
