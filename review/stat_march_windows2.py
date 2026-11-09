import sys, os, numpy as np
sys.path.insert(0, r"C:/Users/user/tfb/src"); os.chdir(r"C:/Users/user/tfb/src")
from cachelib import load, T2_PANELS
from t2_windows import queue_matrix, H, F
tot = [0, 0, 0]
for p in T2_PANELS:
    P = load(p, "validation"); Qm = queue_matrix(P); obs = np.isfinite(P.V) & P.E
    anyq = Qm.any(axis=1); on = []; last = -999
    for t in range(H, P.T - F):
        if P.times[t - H][:10] != P.times[t + F][:10]: continue
        if obs[t - H:t].mean() < 0.4: continue
        if anyq[t - H:t + F].any() or not anyq[t + F]: continue
        if t - last >= 72: on.append(t); last = t
    d = [int(P.times[t][8:10]) for t in on]
    lab_obs = np.mean([obs[t + F].mean() for t in on]) if on else np.nan
    tot[0] += len(on); tot[1] += sum(x >= 10 for x in d)
    print(f"{p:10s} onset-like windows (cov>=0.4) {len(on):3d}, day>=10 {sum(x>=10 for x in d):3d}, observed share of label row {lab_obs:.2f}")
print("total", tot[:2])
