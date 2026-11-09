"""Replication of evidence_blanks.py on two world families (T: A-trained on all train months; U: A6-trained, disjoint
templates/days), morning onset windows only (the official onset windows), release-aligned blanks. Focus on days 1-5
(where the test-month onset windows sit). Variants vs production (cur0.25): none, cur_skip0.25, union0.25.
Pooled cluster bootstrap (panel x world) for days 2-5 and days 1-5.
usage: python evidence_blanks_TU.py"""
import sys, io, contextlib, pickle, numpy as np, pandas as pd
sys.argv = ["x"]
exec(open(r"C:/Users/user/tfb/twin/evidence_blanks.py", encoding="utf-8").read().split("LT = pickle.load")[0])
out = []
for tag, lab, pr in (("T", "twin_labels.pkl", "tw2_nomem"), ("U", "twin_labels_djA.pkl", "djA_nomem")):
    LT = pickle.load(open(f"E:/tfb_nn/{lab}", "rb"))
    with contextlib.redirect_stdout(io.StringIO()):
        out.append(evaluate(LT, blend([W + f"probs_{pr}_s{s}.pkl" for s in (0, 1)]), tag).assign(f=tag))
    WC.clear(); del LT
P = pd.concat(out); rng = np.random.default_rng(5)
print(f"morning onset windows: T {int((P.f == 'T').sum())}, U {int((P.f == 'U').sum())}; visible evidence (days 2-5): {P[(P.day >= 2) & (P.day <= 5)].has_cur.mean():.2f}")
for name, m in (("days 2-5", (P.day >= 2) & (P.day <= 5)), ("days 1-5", P.day <= 5), ("all days", P.day >= 1)):
    S = P[m]; u = S.cl.unique(); idx = {x: np.flatnonzero(S.cl.to_numpy() == x) for x in u}
    msg = []
    for c in ("none", "cur_skip0.25", "union0.25", "after0.25"):
        d = (S[c] - S["cur0.25"]).to_numpy(); se = np.std([d[np.concatenate([idx[x] for x in rng.choice(u, len(u))])].mean() for _ in range(1000)])
        msg.append(f"{c} {d.mean():+.4f} (SE {se:.4f}; T {d[S.f.to_numpy() == 'T'].mean():+.4f} U {d[S.f.to_numpy() == 'U'].mean():+.4f})")
    print(f"{name:9s} n {len(S):4d}: production {S['cur0.25'].mean():.4f} | " + " | ".join(msg))
