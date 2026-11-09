"""A21 (10/24, audit day 3 (statistics), strategy's plan): ledger rows recomputed with an implementation independent of
src/judge.py / m3v2.summary — plain numpy only: per condition the unweighted paired mean difference over windows, a window
bootstrap SE (2,000 draws, seed 0), and T2 = 0.5 onset + 0.5 ongoing (the official windows are 40 + 40 per month).
Compared with the judge's "raw" per-condition values printed in the ledger runs. Hold-out windows: CALENDAR day = 0 mod 3
(0-based day + 1; m3v2.is_holdout). 10/24 00:55: a first run used 0-based day % 3 and so mixed hold-out windows into the
"dev" rows (recorded in AUDIT_20261024; no decision used it). The hold-out row is not recomputed (read once only).
usage: python a21_stats_recompute.py"""
import numpy as np, pandas as pd

W = "C:/Users/user/tfb/work/t2/"
ROWS = [("#29 physics onset on B's onset base (dev)", "g93_final_pair_v3.csv", "B", "phys", "queue_onset", False),
        ("#30-like: A's April parts on the c24f base (dev)", "g93_stack_bv3_c24f_attnglob.csv", "B", "A", None, False),
        ("final pair B -> A on the training-month base (dev)", "g93_final_pair_v3.csv", "B", "A", None, False),
        ("LightGBM on the training-month base (dev)", "g93_final_pair_v3.csv", "B", "lgb", "queue_ongoing", False)]
rng = np.random.default_rng(0)


def boot(d, n=2000):
    d = np.asarray(d, float)
    return float(np.std([d[rng.integers(0, len(d), len(d))].mean() for _ in range(n)]))


for name, f, a, b, cond, hold in ROWS:
    D = pd.read_csv(W + f); D = D[((D.day + 1) % 3 == 0) if hold else ((D.day + 1) % 3 != 0)].dropna(subset=[a, b])
    out = {}
    for c in ("queue_onset", "queue_ongoing"):
        g = D[D.cond == c]; d = (g[b] - g[a]).values
        out[c] = (d.mean(), boot(d), len(d), int((d != 0).sum())) if len(d) else (0.0, 0.0, 0, 0)
    if cond:
        m, se, n, ch = out[cond]; t2 = f"{cond}: {m:+.4f} ± {se:.4f} (n {n}, changed {ch})"
    else:
        m = 0.5 * out["queue_onset"][0] + 0.5 * out["queue_ongoing"][0]; se = 0.5 * np.hypot(out["queue_onset"][1], out["queue_ongoing"][1])
        t2 = (f"T2 (0.5/0.5) {m:+.4f} ± {se:.4f} ({m / se:+.1f} SE) | onset {out['queue_onset'][0]:+.4f} (changed {out['queue_onset'][3]}) "
              f"ongoing {out['queue_ongoing'][0]:+.4f} (changed {out['queue_ongoing'][3]})")
    print(f"{name}: {t2}")
