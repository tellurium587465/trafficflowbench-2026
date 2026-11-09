"""G218 (11/02, ledger #186, stage 0): the selector takes the first origin with persistence IoU <= 0.9, so on an ongoing window the departure from persistence may sit in the last rows of the horizon. Per horizon row h = 1 .. 6 the cells where the truth differs from persistence (the queue at T - 5 held), and the errors of A8's flow (false positives, false negatives): M3 dev, and January, February with B9s' flow; the selector-like windows of all nine training months for the persistence count. Line: the last 2 rows hold >= 60 % of the difference -> stage 1. G175's source with counters; nothing is written. Dev windows only.
usage: TFB_X3=x3af TFB_M3VER=v3 py -B -X utf8 g218_row_shape.py"""
import sys
s = open(r"C:/Users/user/tfb/review/g175_bundle_read.py", encoding="utf-8").read()


def sub(a, b, n=1):
    global s
    assert s.count(a) == n, (a[:60], s.count(a))
    s = s.replace(a, b)


H = '''CNT = {}


def count(tag, sel, y, e, lq, wt):
    c = CNT.setdefault(tag, np.zeros((3, 6))); p_ = np.tile(lq, (6, 1)); c[0] += wt * ((p_ != y) & e).sum(axis=1); c[1] += wt * (sel & ~y & e).sum(axis=1); c[2] += wt * (~sel & y & e).sum(axis=1)


def frame(wins, parts):'''
sub('def frame(wins, parts):', H)
sub('sc["bundle"] += pr * G.iou(dec(tmp((0.8 * ((1 - X3) * (0.8 * xs + 0.2 * pt) + X3 * px) + 0.2 * pl) * cf * sw)), m)', 'sc["bundle"] += pr * G.iou(dec(tmp((0.8 * ((1 - X3) * (0.8 * xs + 0.2 * pt) + X3 * px) + 0.2 * pl) * cf * sw)), m); count("M3", dec(tmp((0.8 * ((1 - X3) * (0.8 * xs + 0.2 * pt) + X3 * px) + 0.2 * pl) * cf * sw)), m["y"].T.astype(bool), m["m"].T.astype(bool), lq, pr)\n            ')
sub('r["B"] += pr * h.iou(h.decode(p_), y, e, "queue_ongoing");', 'count(mo, h.decode(tmp((1 - X3) * p_ + X3 * px)), y, e, lq, pr)\n                    r["B"] += pr * h.iou(h.decode(p_), y, e, "queue_ongoing");')
sub('"g175_dev.csv"', '"g218_dev.csv"', 2); sub('"g175_train_B.csv"', '"g218_train.csv"')
sub('    for mo, d in Df.groupby("month"):', '''    h = module("g119b", R0 + "review/g119_month_postproc.py"); T, GG = h.T, h.G; c = np.zeros(6)
    for k in np.flatnonzero(T.META["is_sel"] & (T.META["cond"] == "queue_ongoing")):
        k = int(k); lab, el = T.LAB[k]; lq = np.asarray(h.XN[h.OFFN[k]:h.OFFN[k + 1]][:, GG.CI], float) == 1; c += ((np.tile(lq, (6, 1)) != lab.astype(bool)) & el.astype(bool)).sum(axis=1)
    print("nine months, windows", int((T.META["is_sel"] & (T.META["cond"] == "queue_ongoing")).sum()), "difference from persistence per row", np.round(c / c.sum(), 3), "last two", round(float(c[-2:].sum() / c.sum()), 3), flush=True)
    for k_, c_ in CNT.items():
        print(k_, "persistence", np.round(c_[0] / c_[0].sum(), 3), "last two", round(float(c_[0, -2:].sum() / c_[0].sum()), 3), "| false pos", np.round(c_[1] / c_[1].sum(), 3), "| false neg", np.round(c_[2] / c_[2].sum(), 3), "| cells", np.round(c_.sum(axis=1), 0), flush=True)
    for mo, d in Df.groupby("month"):''')
sys.argv = [sys.argv[0], "dev"]; exec(compile(s, "g175r", "exec"))
