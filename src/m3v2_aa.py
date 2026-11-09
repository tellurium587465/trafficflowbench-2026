"""M3 v2 validation V1 (METHOD_REVIEW_20261022 §1): A/A tests. For each q24f configuration, the 4 seeds are split into
two 2-seed arms in the 3 possible ways; both arms are scored under M3 v2 (t2_m3v2_gate.score, k expectation, simulated
blanks) and compared with m3v2.summary. Under the null (same model, other seeds) |z| > 2 should happen in about 5% of
comparisons if the SE is right; the share above 2 and the spread of z are reported, with and without the seed-noise term.
Arms are scored in parallel processes.
usage: python m3v2_aa.py [processes=4] [families=q24f]   (10/22: more families -> more A/A pairs, m3v2_aa_<families>.csv)"""
import sys, itertools, pickle
from concurrent.futures import ProcessPoolExecutor
import numpy as np, pandas as pd

FAM = (sys.argv[2] if len(sys.argv) > 2 else "q24f").split(",")
CFG = [f"{f}_{c}" for f in FAM for c in ("attnglob", "attnwp05", "attn1", "drop1", "r13")]
SPLITS = [((0, 1), (2, 3)), ((0, 2), (1, 3)), ((0, 3), (1, 2))]


def score_arm(arg):
    cfg, seeds = arg
    sys.argv = [sys.argv[0], "0"]
    import t2_m3_gate as G, t2_m3v2_gate as V
    ps = [pickle.load(open(G.W + f"probs_{G.PRE}__{cfg}_s{s}.pkl", "rb")) for s in seeds]
    pr = {k: sum(p[k] for p in ps) / len(ps) for k in ps[0]}
    return arg, {w: V.score(pr[w], G.META[w]) for w in G.WID}


if __name__ == "__main__":
    nproc = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    import os
    PRE = "m3" + os.environ.get("TFB_M3VER", "")                # 10/23: window-set version
    CFG = [c for c in CFG if all(os.path.exists(f"C:/Users/user/tfb/work/t2/probs_{PRE}__{c}_s{s}.pkl") for s in range(4))]
    arms = sorted({(c, s) for c in CFG for sp in SPLITS for s in sp})
    with ProcessPoolExecutor(nproc) as ex:
        res = dict(ex.map(score_arm, arms))
    sys.argv = [sys.argv[0], "0"]
    import t2_m3v2_gate as V, m3v2
    rows = []
    for c in CFG:
        for a, b in SPLITS:
            D = V.frame({"a": res[(c, a)], "b": res[(c, b)]})
            for tgt in ("private",):
                s = m3v2.summary(D, "a", "b", tgt)
                rows.append(dict(cfg=c, split=f"{a}v{b}", d=s["t2_weighted"], se=s["se"], z=s["t2_weighted"] / s["se"],
                                 on=s["per_cond"]["queue_onset"]["weighted"], og=s["per_cond"]["queue_ongoing"]["weighted"]))
    R = pd.DataFrame(rows); R.to_csv("C:/Users/user/tfb/work/t2/m3v2_aa" + ("" if FAM == ["q24f"] else "_" + "_".join(FAM)) + os.environ.get("TFB_M3VER", "") + ".csv", index=False)
    print(R.round(4).to_string(index=False))
    sd_d = R.d.std(); mse = np.sqrt((R.se ** 2).mean())
    print(f"A/A: {len(R)} comparisons | share |z| > 2: {np.mean(np.abs(R.z) > 2):.2f} | sd of z {R.z.std():.2f} (1.0 if the SE is right) | "
          f"empirical sd of d {sd_d:.4f} vs mean bootstrap SE {mse:.4f} -> seed-noise sd to add ~ {np.sqrt(max(sd_d ** 2 - mse ** 2, 0)):.4f}")
