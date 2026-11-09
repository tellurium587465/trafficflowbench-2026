"""Write Task 4 candidate frames (validation + private) for LB selection."""
import sys
import numpy as np, pandas as pd
from pathlib import Path
from common import PANELS, WORK
from task4 import load_problem, distinct_rows, solve_weighted_l2, solve_kl, s_link, frame

OUT = Path(WORK) / "t4"; OUT.mkdir(parents=True, exist_ok=True)


def candidates(panel, split):
    P = {s: load_problem(panel, s) for s in ("train", "validation", "private")}
    p = P[split]
    Ar, cr, _, _ = distinct_rows(p["A"], p["c"])
    b = p["b"]
    pooled = np.exp(np.mean([np.log(P[s]["b"]) for s in P], axis=0))
    pooled *= b.sum() / pooled.sum()
    out = {}
    out["l2"] = solve_weighted_l2(Ar, cr, b, np.ones_like(b))[0]
    out["kl"] = solve_kl(Ar, cr, b)[0]
    out["klpool"] = solve_kl(Ar, cr, pooled)[0]
    out["chi2"] = solve_weighted_l2(Ar, cr, b, 1.0 / b)[0]      # (f-b)^2/b
    for k, f in out.items():
        assert np.isfinite(f).all() and (f >= 0).all(), (panel, split, k)
        assert s_link(p, f) > 0.99999, (panel, split, k, s_link(p, f))
    return p, out


if __name__ == "__main__":
    rows = {}
    stats = []
    for split in ("validation", "private"):
        for panel in PANELS:
            p, out = candidates(panel, split)
            for k, f in out.items():
                rows.setdefault(k, []).append(frame(p, f))
                stats.append(dict(panel=panel, split=split, cand=k, sum_f=f.sum(), sum_b=p["b"].sum(),
                                  D=np.abs(f - p["b"]).sum(), zeros=(f < 1e-9).mean()))
    for k, fr in rows.items():
        pd.concat(fr, ignore_index=True).to_parquet(OUT / f"odme_{k}.parquet", index=False)
    s = pd.DataFrame(stats)
    print(s.groupby(["split", "cand"])[["sum_f", "sum_b", "D", "zeros"]].mean().round(3))
