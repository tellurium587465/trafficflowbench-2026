"""C4: simulated stage deltas of the Task 1 pipeline, aggregated like the leaderboard (S_state: family mean of panel
means; S_phys proxy: panel mean; total = 0.35 dS_state + 0.15 dS_phys), next to the deltas the LB showed on 10/8."""
import sys, glob, os, pickle, numpy as np, pandas as pd
sys.path.insert(0, r"C:/Users/user/tfb/src")
from cachelib import FAMILY

SIM = "C:/Users/user/tfb/audit/sim/"
R = {os.path.basename(f)[:-11]: pickle.load(open(f, "rb")) for f in glob.glob(SIM + "*_scores.pkl")}
pans = sorted(R); fams = sorted({FAMILY[p] for p in pans})


def agg(key):
    s = np.mean([np.mean([R[p][key][0] for p in pans if FAMILY[p] == f]) for f in fams])
    ph = np.mean([R[p][key][2] for p in pans])
    return s, ph


steps = [("s17 (f3, no ADAPT)", ("noadapt", "f3 as is"), None),
         ("+gapml", ("noadapt", "+gapml"), 0.0052),
         ("+ADAPT", ("adapt", "+gapml"), 0.0040),
         ("+ceiling clip", ("adapt", "+clip"), 0.00002),
         ("+FD 0.75", ("adapt", "+fd0.75"), 0.0010)]
print("panels:", len(pans), pans)
prev = None
for nm, key, lb in steps:
    s, ph = agg(key)
    if prev is None:
        print(f"{nm:22s} S_state {s:.4f}  S_phys_proxy {ph:.4f}")
    else:
        d = 0.35 * (s - prev[0]) + 0.15 * (ph - prev[1])
        print(f"{nm:22s} S_state {s:.4f}  S_phys_proxy {ph:.4f}  sim dTotal {d:+.4f}  (dS_state {s-prev[0]:+.4f}, dS_phys {ph-prev[1]:+.4f})  LB {lb:+.4f}")
    prev = (s, ph)
base = agg(("adapt", "+fd0.75"))
for nm, key in (("FD 0.5 instead", ("adapt", "+fd0.5")), ("FD 1.0 instead", ("adapt", "+fd1.0")), ("F3 (+FD0.75)", ("adapt_f3", "+fd0.75")),
                ("F3 (+FD0.5)", ("adapt_f3", "+fd0.5"))):
    s, ph = agg(key); d = 0.35 * (s - base[0]) + 0.15 * (ph - base[1])
    better = sum((0.35 * (R[p][key][0] - R[p][("adapt", "+fd0.75")][0]) + 0.15 * (R[p][key][2] - R[p][("adapt", "+fd0.75")][2])) > 0 for p in pans)
    print(f"{nm:22s} vs best: sim dTotal {d:+.5f}  (dS_state {s-base[0]:+.5f}, dS_phys {ph-base[1]:+.5f})  panels better {better}/{len(pans)}")
