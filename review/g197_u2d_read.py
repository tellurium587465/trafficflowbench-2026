"""G197 (11/01, ledger #133): the 2D U-Net (fold 2 weights, G193) on the instruments of G175: A8's flow and B9s' flow against the same flow with 0.2 of the net mixed right after the X3a mix. G175's source with these columns added; G175 itself and its csv are not changed. Dev windows only.
usage: TFB_X3=x3af TFB_M3VER=v3 py -B -X utf8 g197_u2d_read.py"""
import sys
s = open(r"C:/Users/user/tfb/review/g175_bundle_read.py", encoding="utf-8").read()


def sub(a, b, n=1):
    global s
    assert s.count(a) == n, (a[:60], s.count(a))
    s = s.replace(a, b)


sub('CC = ("A", "bundle", "B", "Bc", "B2", "Bc2"); cols', 'PU = [pickle.load(open(W + f"probs_m3v3__u2d_s{s_}.pkl", "rb")) for s_ in (0, 1)]; CC = ("A", "bundle", "B", "Bc", "B2", "Bc2", "bu", "Bu"); cols')
sub('px = (PX[0][w] + PX[1][w]) / 2; pq =', 'px = (PX[0][w] + PX[1][w]) / 2; pu = (PU[0][w] + PU[1][w]) / 2; pq =')
sub('sc["bundle"] += pr * G.iou(dec(tmp((0.8 * ((1 - X3) * (0.8 * xs + 0.2 * pt) + X3 * px) + 0.2 * pl) * cf * sw)), m)', 'sc["bundle"] += pr * G.iou(dec(tmp((0.8 * ((1 - X3) * (0.8 * xs + 0.2 * pt) + X3 * px) + 0.2 * pl) * cf * sw)), m); sc["bu"] += pr * G.iou(dec(tmp((0.8 * (0.8 * ((1 - X3) * (0.8 * xs + 0.2 * pt) + X3 * px) + 0.2 * pu) + 0.2 * pl) * cf * sw)), m)')
sub('sc["Bc"] += pr * G.iou(raw(tmp((1 - X3) * pq + X3 * px)), m);', 'sc["Bc"] += pr * G.iou(raw(tmp((1 - X3) * pq + X3 * px)), m); sc["Bu"] += pr * G.iou(raw(tmp(0.8 * ((1 - X3) * pq + X3 * px) + 0.2 * pu)), m);')
sub('X = [pickle.load(open(f"C:/tfb_data/kaggle_out/cap1/out/oof_x3a_s{s}.pkl", "rb")) for s in (0, 1)]; rows = []', 'X = [pickle.load(open(f"C:/tfb_data/kaggle_out/cap1/out/oof_x3a_s{s}.pkl", "rb")) for s in (0, 1)]; U = [pickle.load(open(W + f"oof_u2d_s{s}.pkl", "rb")) for s in (0, 1)]; rows = []')
sub('r["B"] += pr * h.iou(h.decode(p_), y, e, "queue_ongoing");', 'r["Bu"] = r.get("Bu", 0.0) + pr * h.iou(h.decode(tmp(0.8 * ((1 - X3) * p_ + X3 * px) + 0.2 * (U[0][k] + U[1][k]) / 2)), y, e, "queue_ongoing"); r["B"] += pr * h.iou(h.decode(p_), y, e, "queue_ongoing");')
sub('"g175_dev.csv"', '"g197_dev.csv"', 1 if s.count('"g175_dev.csv"') == 1 else s.count('"g175_dev.csv"')); sub('"g175_train_B.csv"', '"g197_train.csv"')
sub('for a, b, nm in (("A", "bundle", "A side"), ("B", "Bc", "B side"), ("B2", "Bc2", "B side with X1 beta 0.25")):\n        s = m3v2.summary(D, a, b, "private"); sm =', 'for a, b, nm in (("bundle", "bu", "A8 + net"), ("Bc", "Bu", "B9s + net")):\n        s = m3v2.summary(D, a, b, "private"); sm =')
sub('Bc - B {(d.Bc - d.B).mean():+.4f} (SE {boot((d.Bc - d.B).to_numpy(), d.panel + \'_\' + d.day.astype(str)):.4f})', 'Bu - Bc {(d.Bu - d.Bc).mean():+.4f} (SE {boot((d.Bu - d.Bc).to_numpy(), d.panel + \'_\' + d.day.astype(str)):.4f})')
sys.argv = [sys.argv[0], "dev"]; exec(compile(s, "g175x", "exec"))
