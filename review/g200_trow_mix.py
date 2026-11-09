"""G200 (11/01, ledger #164): the row-T networks (fold 2 capBT, seeds 0-3) mixed into A8's ongoing flow, as a smaller step than replacing the 40 production networks. G175's source with these columns: bt<w> = (1 - w) (A8's blend before LightGBM) + w capBT, w in (0.2, 0.35, 0.5). March composition as judge. January, February: the OOF flow of G175's B path, (1 - w) (0.8 q25 + 0.2 X3a) + w capBT with the temperature, against the same without. G175 itself and its csv are not changed. Dev windows only.
usage: TFB_X3=x3af TFB_M3VER=v3 py -B -X utf8 g200_trow_mix.py"""
import sys
s = open(r"C:/Users/user/tfb/review/g175_bundle_read.py", encoding="utf-8").read()


def sub(a, b, n=1):
    global s
    assert s.count(a) == n, (a[:60], s.count(a))
    s = s.replace(a, b)


WS = (0.2, 0.35, 0.5)
sub('CC = ("A", "bundle", "B", "Bc", "B2", "Bc2"); cols', 'PU = [pickle.load(open(W + f"probs_m3v3cap__capBT_s{s_}_fold2030-12.pkl", "rb")) for s_ in range(4)]; CC = ("A", "bundle", "B", "Bc", "B2", "Bc2", "bt0.2", "bt0.35", "bt0.5"); cols')
sub('px = (PX[0][w] + PX[1][w]) / 2; pq =', 'px = (PX[0][w] + PX[1][w]) / 2; pu = sum(d_[w] for d_ in PU) / 4; pq =')
sub('sc["bundle"] += pr * G.iou(dec(tmp((0.8 * ((1 - X3) * (0.8 * xs + 0.2 * pt) + X3 * px) + 0.2 * pl) * cf * sw)), m)', 'sc["bundle"] += pr * G.iou(dec(tmp((0.8 * ((1 - X3) * (0.8 * xs + 0.2 * pt) + X3 * px) + 0.2 * pl) * cf * sw)), m)\n            for w_ in (0.2, 0.35, 0.5):\n                sc[f"bt{w_}"] += pr * G.iou(dec(tmp((0.8 * ((1 - w_) * ((1 - X3) * (0.8 * xs + 0.2 * pt) + X3 * px) + w_ * pu) + 0.2 * pl) * cf * sw)), m)\n            ')
sub('X = [pickle.load(open(f"C:/tfb_data/kaggle_out/cap1/out/oof_x3a_s{s}.pkl", "rb")) for s in (0, 1)]; rows = []', 'X = [pickle.load(open(f"C:/tfb_data/kaggle_out/cap1/out/oof_x3a_s{s}.pkl", "rb")) for s in (0, 1)]; U = [pickle.load(open(f"C:/tfb_data/kaggle_out/trow/ko/oof_capBT_s{s}.pkl", "rb")) for s in range(4)]; rows = []')
sub('r["B"] += pr * h.iou(h.decode(p_), y, e, "queue_ongoing");', 'pu_ = sum(u_[k] for u_ in U) / 4\n                    for w_ in (0.2, 0.35, 0.5):\n                        r[f"bt{w_}"] = r.get(f"bt{w_}", 0.0) + pr * h.iou(h.decode(tmp((1 - w_) * ((1 - X3) * p_ + X3 * px) + w_ * pu_)), y, e, "queue_ongoing")\n                    r["B"] += pr * h.iou(h.decode(p_), y, e, "queue_ongoing");')
sub('"g175_dev.csv"', '"g200_dev.csv"', 2); sub('"g175_train_B.csv"', '"g200_train.csv"')
sub('for a, b, nm in (("A", "bundle", "A side"), ("B", "Bc", "B side"), ("B2", "Bc2", "B side with X1 beta 0.25")):\n        s = m3v2.summary(D, a, b, "private"); sm =', 'for a, b, nm in (("bundle", "bt0.2", "A8 + 0.2"), ("bundle", "bt0.35", "A8 + 0.35"), ("bundle", "bt0.5", "A8 + 0.5")):\n        s = m3v2.summary(D, a, b, "private"); sm =')
sub('print(mo, len(d), f"B raw {d.B.mean():.4f} Bc - B {(d.Bc - d.B).mean():+.4f} (SE {boot((d.Bc - d.B).to_numpy(), d.panel + \'_\' + d.day.astype(str)):.4f})", flush=True)', 'print(mo, len(d), " | ".join(f"w {w_} {(d[f\'bt{w_}\'] - d.Bc).mean():+.4f} (SE {boot((d[f\'bt{w_}\'] - d.Bc).to_numpy(), d.panel + \'_\' + d.day.astype(str)):.4f})" for w_ in (0.2, 0.35, 0.5)), flush=True)')
sys.argv = [sys.argv[0], "dev"]; exec(compile(s, "g175y", "exec"))
