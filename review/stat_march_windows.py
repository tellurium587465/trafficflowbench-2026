"""How many self-made T2 windows can be built from the PUBLIC validation-month (March) observations?
(Only for April-facing evaluation; uses t2_windows.candidates on the released matrices, blanks stay NaN.)"""
import sys, os, numpy as np
sys.path.insert(0, r"C:/Users/user/tfb/src"); os.chdir(r"C:/Users/user/tfb/src")
from cachelib import load, T2_PANELS
from t2_windows import candidates
tot = {"queue_onset": 0, "queue_ongoing": 0}; late = {"queue_onset": 0, "queue_ongoing": 0}
for p in T2_PANELS:
    P = load(p, "validation")
    obs = np.isfinite(P.V) & P.E
    sel, Qm = candidates(P, np.random.default_rng(0), spacing=72)
    days = {c: [int(P.times[t][8:10]) for t in ts] for c, ts in sel.items()}
    for c in tot:
        tot[c] += len(days[c]); late[c] += sum(d >= 10 for d in days[c])
    print(f"{p:10s} observed share {obs.mean():.3f}  blank rows(all links NaN) {np.mean(~np.isfinite(P.V).any(axis=1)):.3f}  "
          f"onset {len(days['queue_onset'])} (day>=10: {sum(d>=10 for d in days['queue_onset'])})  ongoing {len(days['queue_ongoing'])}")
    del P
print("total", tot, "from day 10 on", late)
