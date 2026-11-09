"""G106 (10/24, literature scan 2-5, user go-ahead 10/24 21:5x): a gravity-shaped prior for the Task 4 projection.
The released weak prior b is noisy around a gravity form (log b = origin effect + destination effect + beta x path length:
R^2 0.75, residual sd 0.6, residuals uncorrelated between splits; GENERATOR.md). If the generator's own prior was closer to
the gravity form, shrinking b towards the fitted g in logs before the exact L2 projection moves f towards f*.
Per panel and split: OLS fit of log b on origin / destination dummies + path length (km, from links.csv) -> g;
prior b' = b^alpha g^(1 - alpha), rescaled to the total of b; f = the L2 projection of b' onto {A f = c, f >= 0}.
alpha = 1 is the submitted L2 (identity checked). Pre-declared: alpha = 0.75 (t4g075) and 0.5 (t4g050), one submission each
(Task 4 has no LB noise); line = public LB above finA5's 0.89741. If neither, Task 4 stays L2.
usage: python g106_t4_gravity.py"""
import sys
sys.path.insert(0, r"C:/Users/user/tfb/src")
import numpy as np, pandas as pd
from common import PANELS, WORK, REL
from task4 import load_problem, distinct_rows, solve_weighted_l2, s_link, frame
from submit import assemble


def gravity(P):
    paths = P["paths"]; nd = REL / "corridors" / P["panel"] / "network"
    L = pd.read_csv(nd / "links.csv", dtype={"link_id": str}).set_index("link_id").length_km.to_dict()
    dist = paths.link_seq.astype(str).str.split(";").map(lambda s: sum(L.get(x, 0.0) for x in s)).to_numpy(float)
    o = pd.get_dummies(paths.origin_zone.astype(str), drop_first=False).to_numpy(float)
    d = pd.get_dummies(paths.destination_zone.astype(str), drop_first=True).to_numpy(float)
    X = np.column_stack([o, d, dist]); y = np.log(np.maximum(P["b"], 1e-6)); ok = P["b"] > 0
    beta, *_ = np.linalg.lstsq(X[ok], y[ok], rcond=None); fit = X @ beta
    r2 = 1 - np.var(y[ok] - fit[ok]) / np.var(y[ok])
    return np.exp(fit), r2


out, r2s = {}, []
for a in (1.0, 0.75, 0.5):
    fr = []
    for split in ("validation", "private"):
        for p in PANELS:
            P = load_problem(p, split); Ar, cr, _, _ = distinct_rows(P["A"], P["c"]); b = P["b"]
            if a == 1.0:
                b2 = b
            else:
                g, r2 = gravity(P); r2s.append(r2)
                b2 = np.power(np.maximum(b, 1e-9), a) * np.power(g, 1 - a); b2 *= b.sum() / b2.sum()
            f = solve_weighted_l2(Ar, cr, b2, np.ones_like(b2))[0]
            assert np.isfinite(f).all() and (f >= 0).all() and s_link(P, f) > 0.99999, (p, split, a)
            fr.append(frame(P, f))
    out[a] = pd.concat(fr, ignore_index=True)
ref = pd.read_parquet(WORK + "/t4/odme_l2.parquet")
print("alpha 1 vs odme_l2: max |diff|", float(np.abs(out[1.0].path_flow.to_numpy() - ref.path_flow.to_numpy()).max()), "| gravity fit R^2 median %.2f" % np.median(r2s))
T1 = pd.read_parquet(WORK + "/t1/state_f3ens3c_ad2_gapml_clip_fd_bnd.parquet"); T2 = pd.read_parquet(WORK + "/t2/queue_finA5.parquet")
for a, tag in ((0.75, "g075"), (0.5, "g050")):
    dlt = out[a].path_flow.to_numpy() - ref.path_flow.to_numpy()
    print(f"alpha {a}: |f - f_l2|_1 / |f_l2|_1 = {np.abs(dlt).sum() / ref.path_flow.sum():.4f}")
    out[a].to_parquet(WORK + f"/t4/odme_l2{tag}.parquet", index=False)
    print(assemble(T1, T2, out[a], f"t4{tag}"))
