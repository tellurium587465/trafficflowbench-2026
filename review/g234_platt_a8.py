"""G234 (11/03, ledger #201, plt re-measure): #134's Platt per horizon row read on A8's flow, which has a temperature 1.2 just before the
set (G175's source, src unchanged). Platt = a_h logit p + b_h fitted on the January + February OOF of q25_nomem seeds 3 / 4 as G181
(M3: fitted on both months; January / February: fitted on the other month). Columns: bp = Platt after the temperature, bq = Platt in
place of it (dev, against "bundle"); Bcp, Bcq the same on B9s' flow on January / February (against "Bc"). Then the clean read as G233:
M3 dev's top third of history coverage and January / February merged by inverse variance.
usage: TFB_X3=x3af TFB_M3VER=v3 py -B -X utf8 g234_platt_a8.py"""
import sys
s = open(r"C:/Users/user/tfb/review/g175_bundle_read.py", encoding="utf-8").read()


def sub(a, b, n=1):
    global s
    assert s.count(a) == n, (a[:60], s.count(a))
    s = s.replace(a, b)


sub('def frame(wins, parts):', '''_h = module("g119", R0 + "review/g119_month_postproc.py"); _O = [pickle.load(open(W + f"oof_q25_nomem_s{s_}.pkl", "rb")) for s_ in (3, 4)]


def _fit(ks):
    from sklearn.linear_model import LogisticRegression
    out = {}
    for hh in range(6):
        x, y = [], []
        for k in ks:
            lab, el = _h.T.LAB[k]; e = el[hh].astype(bool); x.append(lg((_O[0][k] + _O[1][k])[hh][e] / 2)); y.append(lab[hh][e].astype(int))
        x, y = np.concatenate(x), np.concatenate(y); m_ = LogisticRegression(C=1e6, max_iter=300).fit(x[:, None], y); out[hh] = (float(m_.coef_[0, 0]), float(m_.intercept_[0]))
    return out


_ALL = [int(k) for k in np.flatnonzero(_h.T.META["is_sel"] & np.isin(_h.MON, _h.MONTHS)) if str(_h.T.META["cond"][k]) == "queue_ongoing"]
PLT = {"all": _fit(_ALL), "2031-01": _fit([k for k in _ALL if _h.MON[k] == "2031-02"]), "2031-02": _fit([k for k in _ALL if _h.MON[k] == "2031-01"])}
print("Platt (all):", {hh: tuple(round(v, 3) for v in PLT["all"][hh]) for hh in range(6)}, flush=True)


def platt(p, pl):
    q = p.copy()
    for hh in range(6):
        a_, b_ = pl[hh]; q[hh] = sg(a_ * lg(p[hh]) + b_)
    return q


def frame(wins, parts):''')
sub('    CC = ("A", "bundle", "B", "Bc", "B2", "Bc2"); cols', '    CC = ("A", "bundle", "B", "Bc", "B2", "Bc2", "bp", "bq"); cols')
sub('sc["bundle"] += pr * G.iou(dec(tmp((0.8 * ((1 - X3) * (0.8 * xs + 0.2 * pt) + X3 * px) + 0.2 * pl) * cf * sw)), m)',
    'xx_ = (0.8 * ((1 - X3) * (0.8 * xs + 0.2 * pt) + X3 * px) + 0.2 * pl) * cf * sw; sc["bundle"] += pr * G.iou(dec(tmp(xx_)), m); sc["bp"] += pr * G.iou(dec(platt(tmp(xx_), PLT["all"])), m); sc["bq"] += pr * G.iou(dec(platt(xx_, PLT["all"])), m)')
sub('r["Bc"] += pr * h.iou(h.decode(tmp((1 - X3) * p_ + X3 * px)), y, e, "queue_ongoing")',
    'xb_ = (1 - X3) * p_ + X3 * px; r["Bc"] += pr * h.iou(h.decode(tmp(xb_)), y, e, "queue_ongoing"); r["Bcp"] = r.get("Bcp", 0.0) + pr * h.iou(h.decode(platt(tmp(xb_), PLT[mo])), y, e, "queue_ongoing"); r["Bcq"] = r.get("Bcq", 0.0) + pr * h.iou(h.decode(platt(xb_, PLT[mo])), y, e, "queue_ongoing")')
sub('D = frame(WIN, "parts"); D.to_csv(W + "g175_dev.csv", index=False)', 'D = frame(WIN, "parts"); D.to_csv(W + "g234_dev.csv", index=False)')
sub('for a, b, nm in (("A", "bundle", "A side"), ("B", "Bc", "B side"), ("B2", "Bc2", "B side with X1 beta 0.25")):\n        s = m3v2.summary(D, a, b, "private"); sm =',
    'for a, b, nm in (("bundle", "bp", "A8 + Platt after the temperature"), ("bundle", "bq", "A8 with Platt in place of the temperature")):\n        s = m3v2.summary(D, a, b, "private"); sm =')
sub('Df = train_B(); Df.to_csv(W + "g175_train_B.csv", index=False)', 'Df = train_B(); Df.to_csv(W + "g234_train.csv", index=False)')
sub("""print(mo, len(d), f"B raw {d.B.mean():.4f} Bc - B {(d.Bc - d.B).mean():+.4f} (SE {boot((d.Bc - d.B).to_numpy(), d.panel + '_' + d.day.astype(str)):.4f})", flush=True)""",
    """print(mo, len(d), f"Bcp - Bc {(d.Bcp - d.Bc).mean():+.4f} (SE {boot((d.Bcp - d.Bc).to_numpy(), d.panel + '_' + d.day.astype(str)):.4f}) | Bcq - Bc {(d.Bcq - d.Bc).mean():+.4f} (SE {boot((d.Bcq - d.Bc).to_numpy(), d.panel + '_' + d.day.astype(str)):.4f})", flush=True)""")
sys.argv = [sys.argv[0], "dev"]
exec(compile(s, "g175p", "exec"))
