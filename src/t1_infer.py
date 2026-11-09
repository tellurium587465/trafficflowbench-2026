"""Predict Task 1 for validation + private with a trained model pair; write state parquet."""
from __future__ import annotations
import sys, json
from pathlib import Path
import numpy as np, pandas as pd, lightgbm as lgb
from cachelib import load
from common import PANELS
from t1_features import grid_features, rows
from t1_train import base_of

MD = Path(r"C:/Users/user/tfb/work/t1")


def predict_panel(panel, split, mv, mq, pi):
    P = load(panel, split)
    F, V, Q, ok = grid_features(P)
    bv, bq = base_of(F)
    r, c = np.nonzero(P.target)
    X, names = rows(P, F, r, c)
    X = np.column_stack([np.full(len(r), pi, np.float32), X])
    v = (bv[r, c] + mv.predict(X, num_threads=16)) * P.vf[c]
    q = (bq[r, c] + mq.predict(X, num_threads=16)) * P.lanes[c]
    v = np.clip(v, 3.0, 1.15 * P.vf[c]); q = np.clip(q, 60.0, None)
    regs = np.array(["R1", "R2", "R3"])[P.regime[r]]
    return pd.DataFrame({"panel": panel, "timestamp": P.times[r], "link_id": P.links[c], "mask_regime": regs,
                         "speed_kmh": v, "flow_vph": q})


if __name__ == "__main__":
    tag = sys.argv[1] if len(sys.argv) > 1 else "v1"
    mv = lgb.Booster(model_file=str(MD / "speed.txt")); mq = lgb.Booster(model_file=str(MD / "flow.txt"))
    parts = []
    for split in ("validation", "private"):
        for pi, panel in enumerate(PANELS):
            parts.append(predict_panel(panel, split, mv, mq, pi)); print(panel, split, len(parts[-1]), flush=True)
    out = pd.concat(parts, ignore_index=True)
    out.to_parquet(MD / f"state_{tag}.parquet", index=False)
    print("rows", len(out))
