"""G206 (11/02, ledger #162): the tail of a queue moved by the shock speed. For each queue at T - 5 (segments of the links queued, gaps up to 2): q_head = flow just past the head, q_in, v_in = flow and speed just upstream of the tail, rows T - 15 .. T - 5; shock speed w = (q_in - q_head) / (k_j - k_in) with k_in = q_in / v_in. At T + 5 h the predicted tail is w x 5 h / 60 km upstream of the tail. Growing: links between the tail and the predicted tail x 1.5. Shrinking: links of the queue between the tail and the predicted tail x 0.67. Arms: grow only, shrink only, both, on A8's blend before the cone; B9s' flow for January, February. G175's source with these columns; G175 and its csv are not changed. Dev windows only.
usage: TFB_X3=x3af TFB_M3VER=v3 py -B -X utf8 g206_shock.py"""
import sys
s = open(r"C:/Users/user/tfb/review/g175_bundle_read.py", encoding="utf-8").read()


def sub(a, b, n=1):
    global s
    assert s.count(a) == n, (a[:60], s.count(a))
    s = s.replace(a, b)


H = '''def shock(P, lq, t):
    n = P.n; Mg = np.ones((6, n)); Ms = np.ones((6, n)); l = np.flatnonzero(lq); x = np.concatenate([[0.0], np.cumsum(P.length[:-1])])
    if not len(l):
        return Mg, Ms
    with np.errstate(all="ignore"):
        for seg in np.split(l, np.flatnonzero(np.diff(l) > 3) + 1):
            u, h_ = int(seg[0]), int(seg[-1]); up = np.arange(max(u - 2, 0), u)
            if not len(up) or h_ + 1 >= n:
                continue
            qh = np.nanmean(P.Q[t - 3:t, h_ + 1]); qi = np.nanmean(P.Q[t - 3:t][:, up]); vi = np.nanmean(P.V[t - 3:t][:, up])
            if not (np.isfinite(qh) and np.isfinite(qi) and np.isfinite(vi)):
                continue
            ki = qi / max(vi, 1.0); w_ = (qi - qh) / max(P.kj[u] - ki, 1e-3)
            for j in range(6):
                d = w_ * 5 * (j + 1) / 60
                if d > 0:
                    Mg[j, (x >= x[u] - d) & (x < x[u])] = 1.5
                elif d < 0:
                    m_ = (x >= x[u]) & (x < x[u] - d); m_[h_ + 1:] = False; Ms[j, m_] = 0.67
    return Mg, Ms


def frame(wins, parts):'''
sub('def frame(wins, parts):', H)
sub('CC = ("A", "bundle", "B", "Bc", "B2", "Bc2"); cols', 'CC = ("A", "bundle", "B", "Bc", "B2", "Bc2", "sg", "ss", "sb"); cols')
sub('sc["bundle"] += pr * G.iou(dec(tmp((0.8 * ((1 - X3) * (0.8 * xs + 0.2 * pt) + X3 * px) + 0.2 * pl) * cf * sw)), m)', 'sc["bundle"] += pr * G.iou(dec(tmp((0.8 * ((1 - X3) * (0.8 * xs + 0.2 * pt) + X3 * px) + 0.2 * pl) * cf * sw)), m); bl_ = 0.8 * ((1 - X3) * (0.8 * xs + 0.2 * pt) + X3 * px) + 0.2 * pl; Mg, Ms = shock(P, lq, t)\n            for c_n, M_ in (("sg", Mg), ("ss", Ms), ("sb", Mg * Ms)):\n                sc[c_n] += pr * G.iou(dec(tmp(np.minimum(bl_ * M_, 1 - 1e-6) * cf * sw)), m)\n            ')
sub('r["B"] += pr * h.iou(h.decode(p_), y, e, "queue_ongoing");', 'bl_ = (1 - X3) * p_ + X3 * px; Mg, Ms = shock(P, lq, t)\n                    for c_n, M_ in (("sg", Mg), ("ss", Ms), ("sb", Mg * Ms)):\n                        r[c_n] = r.get(c_n, 0.0) + pr * h.iou(h.decode(tmp(np.minimum(bl_ * M_, 1 - 1e-6))), y, e, "queue_ongoing")\n                    r["B"] += pr * h.iou(h.decode(p_), y, e, "queue_ongoing");')
sub('"g175_dev.csv"', '"g206_dev.csv"', 2); sub('"g175_train_B.csv"', '"g206_train.csv"')
sub('for a, b, nm in (("A", "bundle", "A side"), ("B", "Bc", "B side"), ("B2", "Bc2", "B side with X1 beta 0.25")):\n        s = m3v2.summary(D, a, b, "private"); sm =', 'for a, b, nm in (("bundle", "sg", "grow"), ("bundle", "ss", "shrink"), ("bundle", "sb", "both")):\n        s = m3v2.summary(D, a, b, "private"); sm =')
sub('print(mo, len(d), f"B raw {d.B.mean():.4f} Bc - B {(d.Bc - d.B).mean():+.4f} (SE {boot((d.Bc - d.B).to_numpy(), d.panel + \'_\' + d.day.astype(str)):.4f})", flush=True)', 'print(mo, len(d), " | ".join(f"{c_} {(d[c_] - d.Bc).mean():+.4f} (SE {boot((d[c_] - d.Bc).to_numpy(), d.panel + \'_\' + d.day.astype(str)):.4f})" for c_ in ("sg", "ss", "sb")), flush=True)')
sys.argv = [sys.argv[0], "dev"]; exec(compile(s, "g175s", "exec"))
