"""Are 'rare-location' onsets random incidents or flow/capacity-driven breakdowns?
For every train onset window: percentile (within the corridor) of hazard proxies at the links that queue at T+30,
split by whether the location is one of the panel's top-3 recurrent onset links."""
import os, pickle, numpy as np
from collections import Counter
C = os.environ.get("TFB_T2CACHE", "2_noT")
Z = np.load(rf"C:/Users/user/tfb/work/t2/featcache{C}.npz", allow_pickle=True)
off, panel, cond, date = Z["off"], Z["panel"], Z["cond"], Z["date"]; names = list(Z["names"]); X = Z["X"]
LAB = pickle.load(open(rf"C:/Users/user/tfb/work/t2/labels_all{C}.pkl", "rb"))
ARR = np.load(rf"C:/Users/user/tfb/work/t2/arrcache{C}.npy", mmap_mode="r")
iq, icap, ilan = names.index("last_q"), names.index("capl"), names.index("lanes")
ks = np.flatnonzero(cond == "queue_onset")
heads = {}
for k in ks:                                         # most downstream queued link at T+30 = the bottleneck head
    y = LAB[k][0][-1].astype(bool)
    if y.any():
        heads[k] = int(np.flatnonzero(y).max())
top = {p: {l for l, _ in Counter(h for k, h in heads.items() if panel[k] == p).most_common(3)} for p in set(panel[ks])}
res = {"recurrent": [[], [], []], "rare": [[], [], []]}
for k, h in heads.items():
    a, b = off[k], off[k + 1]
    qc = X[a:b, iq] / np.maximum(X[a:b, icap], 1e-6)
    arr = np.asarray(ARR[a:b, 6]); arr20 = np.asarray(ARR[a:b, 1])
    grp = "recurrent" if h in top[panel[k]] else "rare"
    for j, s in enumerate((qc, arr, arr20)):
        s = np.nan_to_num(s, nan=-1)
        res[grp][j].append((s < s[h]).mean() + 0.5 * (s == s[h]).mean())   # percentile of the head link
for g, v in res.items():
    v = [np.array(x) for x in v]
    print(f"{g:9s} n={len(v[0]):5d}  pct(q/cap) median {np.median(v[0]):.3f} top10% {np.mean(v[0] >= 0.9):.3f} | "
          f"pct(arr30/dnmin) median {np.median(v[1]):.3f} top10% {np.mean(v[1] >= 0.9):.3f} | pct(arr20) median {np.median(v[2]):.3f}")
