"""G216 (11/02, ledger #184, stage 0): inside a queue the speed follows the congested branch at the flow, so a link is labelled queued only while its flow is below q* = w k_j / (1 + w / (0.6 v_f)). Of the cells of A8's flow predicted queued but free at T + h on links queued at T - 5 (the loss at the end of a queue): the share with the flow at T + h >= q*, and among those the share queued again on the same link within the next hour (the rectangle went on). January, February on B9s' flow. Line (declared first): share >= 15 % -> stage 1. G175's source with counters; nothing is written. Dev windows only.
usage: TFB_X3=x3af TFB_M3VER=v3 py -B -X utf8 g216_qstar.py"""
import sys
s = open(r"C:/Users/user/tfb/review/g175_bundle_read.py", encoding="utf-8").read()


def sub(a, b, n=1):
    global s
    assert s.count(a) == n, (a[:60], s.count(a))
    s = s.replace(a, b)


H = '''CNT = {}


def count(tag, sel, y, e, lq, P, t, wt, q_):
    qs = P.w * P.kj / (1 + P.w / (0.6 * P.vf)); fp = sel & ~y & e & lq[None, :]; c = CNT.setdefault(tag, dict(fp=0.0, fq=0.0, fb=0.0, fn=0.0, allfp=0.0)); c["allfp"] += wt * float((sel & ~y & e).sum()); c["fn"] += wt * float((~sel & y & e).sum()); c["fp"] += wt * float(fp.sum())
    for h_, j_ in zip(*np.nonzero(fp)):
        r_ = t + 1 + h_
        if r_ < len(P.Q) and np.isfinite(P.Q[r_, j_]) and P.Q[r_, j_] >= qs[j_]:
            c["fq"] += wt
            if q_[r_ + 1:r_ + 13, j_].any():
                c["fb"] += wt


def frame(wins, parts):'''
sub('def frame(wins, parts):', H)
sub('sc["bundle"] += pr * G.iou(dec(tmp((0.8 * ((1 - X3) * (0.8 * xs + 0.2 * pt) + X3 * px) + 0.2 * pl) * cf * sw)), m)', 'sc["bundle"] += pr * G.iou(dec(tmp((0.8 * ((1 - X3) * (0.8 * xs + 0.2 * pt) + X3 * px) + 0.2 * pl) * cf * sw)), m); count("M3", dec(tmp((0.8 * ((1 - X3) * (0.8 * xs + 0.2 * pt) + X3 * px) + 0.2 * pl) * cf * sw)), m["y"].T.astype(bool), m["m"].T.astype(bool), lq, P, t, pr, q)\n            ')
sub('r["B"] += pr * h.iou(h.decode(p_), y, e, "queue_ongoing");', 'count(mo, h.decode(tmp((1 - X3) * p_ + X3 * px)), y, e, lq, P, t, pr, c_.q)\n                    r["B"] += pr * h.iou(h.decode(p_), y, e, "queue_ongoing");')
sub('"g175_dev.csv"', '"g216_dev.csv"', 2); sub('"g175_train_B.csv"', '"g216_train.csv"')
sub('    for mo, d in Df.groupby("month"):', '    for k_, c in CNT.items():\n        print(k_, {a: round(b, 1) for a, b in c.items()}, "share of the end loss with flow >= q*", round(c["fq"] / max(c["fp"], 1e-9), 3), "and queued again", round(c["fb"] / max(c["fp"], 1e-9), 3), "share of all false positives on those links", round(c["fp"] / max(c["allfp"], 1e-9), 3), flush=True)\n    for mo, d in Df.groupby("month"):')
sys.argv = [sys.argv[0], "dev"]; exec(compile(s, "g175q", "exec"))
