"""G353 (11/05, ledger #277): calibrate the T+30 row only, on the 80 official June windows (ongoing 40). Fit: logistic slope + intercept on
logit(q) of the T+30 row's scored cells (q = what the decode sees, temperature 0.8, cvf fold 2030-06; the k draw of the largest weight from
G347's dump), leave one panel out (the panel's windows read with the other panels' fit). Read: G328's June ongoing flow on the 80 official
windows (G342b's augmented evaluator), k expectation, the T+30 row of q replaced by sigmoid(a logit(q) + b) of the window's panel fold,
then best_set and the far gate as the base. Line (#277): points > 0 and z >= +1 (cluster bootstrap panel x day), then the LB pair.
usage: TFB_M3VER=v3 py -B -X utf8 g353_t30_calib.py"""
import os, sys, pickle
R0 = "C:/Users/user/tfb/"; W = R0 + "work/t2/"
g = open(R0 + "review/g342b_jun80_run.py", encoding="utf-8").read(); g = g[:g.index("src = open(")]; sys.argv = ["g342b", "none"]; exec(compile(g, "g342b_head", "exec")); sys.argv = sys.argv[:1]
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
lgt = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
DG = pickle.load(open(W + "g347_og_jun80.pkl", "rb")); cells = []
for k, v in DG.items():
    if isinstance(k, str) or k < A["NWIN"]:
        continue
    pr, kk, q, st, lq, y, e, p, t, day, split = v; m_ = e[5].astype(bool)
    cells += [(p, x, int(yy)) for x, yy in zip(lgt(q[5][m_]), y[5][m_])]
C = pd.DataFrame(cells, columns=["panel", "z", "y"]); AB = {}
for p in sorted(C.panel.unique()):
    d = C[C.panel != p]; lr = LogisticRegression(C=1e6, max_iter=1000).fit(d[["z"]], d.y); AB[p] = (float(lr.coef_[0][0]), float(lr.intercept_[0]))
print("T+30 cells", len(C), "positive share", round(C.y.mean(), 3), "| fold (a, b):", {p: tuple(round(x, 3) for x in ab) for p, ab in AB.items()}, flush=True)
s = open(R0 + "review/g328_oracle_map.py", encoding="utf-8").read()


def sub(a, b, n=1):
    global s
    assert s.count(a) == n, (a[:70], s.count(a))
    s = s.replace(a, b)


s = s.replace("junmh0", "jun80").replace("g321_jun_", "g342_jun80_").replace("g329_jun_onset_rows", "g342_jun80_onset_rows")
sub('(("January / February", ("2031-01", "2031-02")), ("June 2030", ("2030-06",)))', '(("June 2030", ("2030-06",)),)')
sub('OG = ("O_box", "O_tail", "O_new", "O_box+tail", "O_map", "O_map_any");', 'OG = ("cal",);')
sub('        for nm, q in og_variants(p, y, sets, mp, mpa).items():\n            r[nm] += pr * G.iou(g132.gate(bs(tmp(q, 0.8)), ex, nr), m)\n', '')
sub('                sets = og_sets(lq); mp, mpa = maps(Hm, (t - r0) // 288, t % 288); yt = y & e; r = dict(w=int(k), base=0.0, **dict.fromkeys(OG, 0.0))',
    '                sets = og_sets(lq); mp, mpa = maps(Hm, (t - r0) // 288, t % 288); yt = y & e; r = dict(w=int(k), panel=p, day=dom, base=0.0, **dict.fromkeys(OG, 0.0))')
sub('                    for nm, q in og_variants(pb, yt, sets, mp, mpa).items():\n                        r[nm] += pr * h.iou(h.far_gate(c, h.decode(tmp(q, 0.8)), t, kk, lq), y, e, "queue_ongoing")\n',
    '                    q2 = tmp(pb, 0.8).copy(); a_, b_ = AB[p]; q2[5] = 1 / (1 + np.exp(-(a_ * LGT(q2[5]) + b_))); r["cal"] += pr * h.iou(h.far_gate(c, h.decode(q2), t, kk, lq), y, e, "queue_ongoing")\n')
sub("    D = pd.DataFrame(rows); b = D.base;", "    D = pd.DataFrame(rows); D.to_csv(W + 'g353_' + lab.split(' (')[0].replace(' / ', '_').replace(' ', '_') + '.csv', index=False); b = D.base;")
s += '''
d = pd.read_csv(W + "g353_June_2030_ongoing.csv"); x = (d.cal - d.base).to_numpy(); cl = pd.factorize(d.panel.astype(str) + "_" + d.day.astype(str))[0]
Rm = np.random.default_rng(0).multinomial(cl.max() + 1, np.full(cl.max() + 1, 1 / (cl.max() + 1)), size=2000); se = float(((Rm @ np.bincount(cl, x)) / (Rm @ np.bincount(cl))).std())
print(f"#277 official June ongoing: cal - base {x.mean():+.5f} (SE {se:.5f}, z {x.mean() / max(se, 1e-12):+.2f}, changed {int((np.abs(x) > 1e-12).sum())}); line (> 0 and z >= +1): {bool(x.mean() > 0 and x.mean() / max(se, 1e-12) >= 1)}", flush=True)
'''
LGT = lgt; exec(compile(s, "g328_g353", "exec"))
