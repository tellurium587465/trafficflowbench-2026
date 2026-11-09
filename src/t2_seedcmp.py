"""Seed-aware comparison of two NN configurations (R1 audit priority 1). Each side is several seeds of one config;
seed means are compared with SE_total = sqrt(SE_boot^2 + sigma_seed^2 (1/n_a + 1/n_b)). sigma_seed was measured on
10/08 from three seeds of the attention + global-context config (EXPERIMENTS.md). Also the stress test: the drop from
the clean OOF to the free-flow-redraw OOFs <tag>_j1 / <tag>_j3 (written by t2_nn2.py with EJIT=1+3).

usage: python t2_seedcmp.py <a_s3,a_s4,...> <b_s3,b_s4,...>     (b is judged against a)
"""
import os, sys, numpy as np
import t2_eval_surprise as S
from t2_eval_oof import evaluate

SIG = dict(vlb=0.0023, on=0.0039, og=0.0014, plain=0.0035, wtd=0.0038, sub=0.0051)   # single-run seed sd
OOF = "C:/Users/user/tfb/work/t2/oof_{}.pkl"
A, B = (s.split(",") for s in sys.argv[1:3])


def metrics(tag):
    v = evaluate(tag)[2]; r = S.per_window(tag)
    return dict(vlb=v[0], on=v[1], og=v[2], **dict(zip(("plain", "wtd", "sub"), S.stats(r)))), r


def side(tags):
    M, R = zip(*(metrics(t) for t in tags))
    return {k: np.mean([m[k] for m in M]) for k in SIG}, np.nanmean(np.vstack(R), axis=0), M


ma, ra, Ma = side(A); mb, rb, Mb = side(B)
se_b = dict(zip(("plain", "wtd", "sub"), S.boot_se(rb - ra)))
print(f"A = {A}\nB = {B}")
for k in SIG:
    d = mb[k] - ma[k]; se = np.sqrt(se_b.get(k, 0.0) ** 2 + SIG[k] ** 2 * (1 / len(A) + 1 / len(B)))
    print(f"  {k:6s} A {ma[k]:.4f}  B {mb[k]:.4f}  diff {d:+.4f}  SE_total {se:.4f}  z {d / se:+.1f}"
          f"   (per seed A {' '.join(f'{m[k]:.4f}' for m in Ma)} | B {' '.join(f'{m[k]:.4f}' for m in Mb)})")

for j in ("j1", "j3"):                                          # stress test: how much does each side lose?
    if not all(os.path.exists(OOF.format(f"{t}_{j}")) for t in A + B):
        continue
    da = side([f"{t}_{j}" for t in A])[0]; db = side([f"{t}_{j}" for t in B])[0]
    print(f"stress {j}: drop A " + " ".join(f"{k} {da[k] - ma[k]:+.4f}" for k in ("vlb", "on", "og", "wtd")) +
          " | drop B " + " ".join(f"{k} {db[k] - mb[k]:+.4f}" for k in ("vlb", "on", "og", "wtd")))
