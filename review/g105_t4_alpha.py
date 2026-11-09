"""G105 (10/24, strategy X item 2): a one-parameter probe of the Task 4 projection's distance. The submitted odme_l2 is the
Euclidean projection of the weak prior b onto {A f = c, f >= 0} (LB S_ODME ~0.994); the chi^2 projection (weights 1/b)
scored ~0.862. Weighted projections min 0.5 sum w (f - b)^2 with w = (b / mean b)^(-alpha) interpolate the two (alpha = 0:
L2, 1: chi^2); alpha = +0.1 and -0.1 probe the slope on both sides of the current choice. Task 4 on the LB has no noise
(deterministic against f*), so one submission each answers it. Identity: alpha = 0 reproduces odme_l2.parquet.
Writes work/t4/odme_l2a<tag>.parquet and the zips (T1 / T2 of finA4 unchanged) work/subs/t4a<tag>_final.zip.
usage: python g105_t4_alpha.py"""
import sys
sys.path.insert(0, r"C:/Users/user/tfb/src")
import numpy as np, pandas as pd
from common import PANELS, WORK
from task4 import load_problem, distinct_rows, solve_weighted_l2, s_link, frame
from submit import assemble

out = {}
for a in (0.0, 0.1, -0.1):
    fr = []
    for split in ("validation", "private"):
        for p in PANELS:
            P = load_problem(p, split); Ar, cr, _, _ = distinct_rows(P["A"], P["c"]); b = P["b"]
            f = solve_weighted_l2(Ar, cr, b, (b / b.mean()) ** (-a))[0]
            assert np.isfinite(f).all() and (f >= 0).all() and s_link(P, f) > 0.99999, (p, split, a)
            fr.append(frame(P, f))
    out[a] = pd.concat(fr, ignore_index=True)
ref = pd.read_parquet(WORK + "/t4/odme_l2.parquet")
print("alpha 0 vs odme_l2: max |diff|", float(np.abs(out[0.0].path_flow.to_numpy() - ref.path_flow.to_numpy()).max()))
T1 = pd.read_parquet(WORK + "/t1/state_f3ens3c_ad2_gapml_clip_fd_bnd.parquet"); T2 = pd.read_parquet(WORK + "/t2/queue_finA4.parquet")
for a, tag in ((0.1, "p010"), (-0.1, "m010")):
    d = out[a].path_flow.to_numpy() - ref.path_flow.to_numpy()
    print(f"alpha {a:+.1f}: |f - f_l2|_1 / |f_l2|_1 = {np.abs(d).sum() / ref.path_flow.sum():.4f}, zeros {np.mean(out[a].path_flow < 1e-9):.3f} (L2 {np.mean(ref.path_flow < 1e-9):.3f})")
    out[a].to_parquet(WORK + f"/t4/odme_l2a{tag}.parquet", index=False)
    print(assemble(T1, T2, out[a], f"t4a{tag}"))
