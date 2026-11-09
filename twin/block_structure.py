"""Does the real generator compute congestion on blocks coarser than the released links? (10/14, A4) D7_I10_E episodes
appear on 6 short links at once at one speed and keep that exact extent for 90 min - not how a queue grows on a fine
grid. If the generator's cells are blocks of links, adjacent links inside a block switch between free and congested in
the same row (they share one state) and congestion edges only sit on block boundaries.
Per panel (raw C: data): for every adjacent pair (i, i+1) with both observed, the split rate = P(exactly one congested |
at least one congested); then blocks = maximal runs of pairs with split rate <= 0.05 (over pairs seen congested >= 30
rows). Also the speed ratio v_i / v_{i+1} inside congested rows (one shared state -> only the 0.8% noise) and the
length of each block. Writes work/twin_blocks.csv (one row per adjacent pair).
usage: python block_structure.py [panels=ALL T2 panels]"""
import sys, glob, numpy as np, pandas as pd, warnings
warnings.simplefilter("ignore")
sys.path.insert(0, r"C:/Users/user/tfb/src"); sys.path.insert(0, r"C:/Users/user/tfb/twin")
from cachelib import load, T2_PANELS
from onset_shape3_lib import raw_matrices

out = []
for p in (sys.argv[1].split(",") if len(sys.argv) > 1 else T2_PANELS):
    P = load(p, "train"); V, Q, E = raw_matrices(P); L = np.nan_to_num(P.length, nan=0.0)
    ok = np.isfinite(V); cong = (V <= P.vcut[None, :]) & ok
    a, b = cong[:, :-1], cong[:, 1:]; both = ok[:, :-1] & ok[:, 1:]
    anyc = (a | b) & both; one = (a ^ b) & both
    n_any = anyc.sum(0); split = np.where(n_any > 0, one.sum(0) / np.maximum(n_any, 1), np.nan)
    bc = a & b & both
    lr = np.where(bc, np.abs(np.log(np.maximum(V[:, :-1], 0.1) / np.maximum(V[:, 1:], 0.1))), np.nan)
    med_lr = np.nanmedian(lr, axis=0)
    for i in range(P.n - 1):
        out.append(dict(panel=p, i=i, len_i=round(float(L[i]), 3), n_cong=int(n_any[i]), split=round(float(split[i]), 3) if n_any[i] else np.nan,
                        abs_log_vratio=round(float(med_lr[i]), 4) if np.isfinite(med_lr[i]) else np.nan))
    D = pd.DataFrame([r for r in out if r["panel"] == p]); busy = D[D.n_cong >= 30]
    tight = busy[busy.split <= 0.05]
    # blocks: runs of consecutive tight pairs
    blocks = []
    for i in tight.i:
        if blocks and blocks[-1][1] == i:
            blocks[-1][1] = i + 1
        else:
            blocks.append([i, i + 1])
    bl = [(s, e, round(float(L[s:e + 1].sum()), 2)) for s, e in blocks]
    print(f"{p}: pairs seen congested>=30: {len(busy)}, split<=0.05: {len(tight)} ({len(tight) / max(len(busy), 1):.0%}), "
          f"split q10/50/90 {np.round(busy.split.quantile([.1, .5, .9]).to_numpy(), 3).tolist()}, "
          f"|log v ratio| inside tight pairs {tight.abs_log_vratio.median():.4f} vs others {busy[busy.split > 0.05].abs_log_vratio.median():.4f}", flush=True)
    print(f"   blocks (first link, last link, km): {bl[:12]}", flush=True)
    load.cache_clear(); del V, Q, E
pd.DataFrame(out).to_csv(r"C:/Users/user/tfb/work/twin_blocks.csv", index=False)
