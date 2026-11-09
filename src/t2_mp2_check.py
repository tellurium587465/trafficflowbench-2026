"""Month-start selector onset windows (train, all months x panels): does argmax of the month prior hit the true
onset head (most downstream queued link at T+30, +-2 links)? v2 (blanks reproduced, head-based) vs v1."""
import numpy as np
from cachelib import load, T2_PANELS
from t2_windows import queue_matrix
from t2_selector import scan
import t2_monthprior as M1
import t2_monthprior2 as M2

hit = {"v1_first": [], "v2_head": [], "v2_any": [], "v2_fhead": [], "v2_thead": [], "v2_tod": []}; nan_days = 0; n = 0
for p in T2_PANELS:
    P = load(p, "train"); Qm = queue_matrix(P); mon = np.array([s[:7] for s in P.times])
    for m in np.unique(mon):
        r0 = np.flatnonzero(mon == m)[0]
        on, _ = scan(P, Qm, start=r0 + 12)
        for t in on:
            y = Qm[t + 6]
            if not y.any():
                continue
            h = int(np.flatnonzero(y).max()); n += 1
            f1 = M1.features(p, "train", t); f2 = M2.features(p, "train", t)
            if f2[0, 5] == 0:
                nan_days += 1
                continue
            for k, v in (("v1_first", f1[:, 2]), ("v2_head", f2[:, 2]), ("v2_any", f2[:, 0]), ("v2_fhead", f2[:, 3]), ("v2_thead", f2[:, 4]), ("v2_tod", f2[:, 1])):
                hit[k].append(abs(int(np.nanargmax(np.nan_to_num(v, nan=-1))) - h) <= 2)
    print(p, "done", flush=True)
print(f"onset windows {n}, day-1 (no prior) {nan_days}")
for k, v in hit.items():
    print(f"  argmax({k}) within +-2 links of the true head: {np.mean(v):.3f} (n={len(v)})")
