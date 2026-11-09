"""Task 1 transfer measured in the digital twin (MAP.md 5-5, 10/12): is the twin an instrument for the T1 / T3 gap
between the local February check and the LB (S_LWR proxy 0.62-0.66 locally vs ~0.46 on the LB)?
One saved model pair (default f3big, feature version 3, production flags TFB_T1_ADAPT=1 TFB_T1_FEAT_TRAINSTAT=1 set
here) scored with metrics.state_score / physics_score on the target cells of
  real : the real train split, February only (held out from f3big's fit; ADAPT off there, as the split is 'train')
  A    : twin world A, February only (the real February replayed by the CTM; same configuration -> calibration)
  T<k> : twin test worlds (new scenario seed: moved bottleneck map, capacity / free-speed / demand shifts)
If A ~ real and T << A by about the LB gap, the twin measures T1 transfer.
usage: python t1_twin_eval.py [tag=f3big] [worlds=T0,T1,T2,T3]"""
import os, sys
os.environ.setdefault("TFB_T1_ADAPT", "1"); os.environ.setdefault("TFB_T1_FEAT_TRAINSTAT", "1")
import numpy as np, lightgbm as lgb
from pathlib import Path
from cachelib import load
from common import PANELS
from t1_exp import featfns
from metrics import state_score, physics_score

MD = Path(r"C:/Users/user/tfb/work/t1")
tag = sys.argv[1] if len(sys.argv) > 1 else "f3big"
worlds = (sys.argv[2] if len(sys.argv) > 2 else "T0,T1,T2,T3").split(",")
gf, rf, bf = featfns(3)
mv = lgb.Booster(model_file=str(MD / f"speed_{tag}.txt")); mq = lgb.Booster(model_file=str(MD / f"flow_{tag}.txt"))


def score(P, pi, keep_rows=None):
    F, _, _, _ = gf(P); bv, bq = bf(F)
    tgt = P.target & P.E & np.isfinite(P.V) & np.isfinite(P.Q)
    if keep_rows is not None:
        tgt &= keep_rows[:, None]
    r, c = np.nonzero(tgt)
    X, _ = rf(P, F, r, c); X = np.column_stack([np.full(len(r), pi, np.float32), X])
    Vp = np.full((P.T, P.n), np.nan); Qp = Vp.copy()
    Vp[r, c] = np.clip((bv[r, c] + mv.predict(X, num_threads=16)) * P.vf[c], 3.0, 1.15 * P.vf[c])
    Qp[r, c] = np.clip((bq[r, c] + mq.predict(X, num_threads=16)) * P.lanes[c], 60.0, None)
    tg, rg = P.target.copy(), P.regime.copy()
    if keep_rows is not None:
        P.target = P.target & keep_rows[:, None]; P.regime = np.where(keep_rows, P.regime, 255).astype(np.uint8)
    s = state_score(P, Vp, Qp); ph = physics_score(P, Vp, Qp)
    P.target, P.regime = tg, rg
    return s, ph[0], ph[1], ph[2], len(r)


rows = []
for pi, panel in enumerate(PANELS):
    if not (Path(r"C:/Users/user/tfb/work/cache") / f"{panel}_twinT0.npz").exists():
        continue
    for w in ["real", "A"] + worlds:
        P = load(panel, "train" if w == "real" else f"twin{w}")
        feb = np.array([str(s).startswith("2031-02") for s in P.times]) if w in ("real", "A") else None
        s, sfd, slwr, sph, n = score(P, pi, feb)
        rows.append((panel, w, s, sfd, slwr, sph, n))
        print(f"{panel} {w:5s} S_state {s:.4f}  S_FD {sfd:.4f}  S_LWR(proxy) {slwr:.4f}  S_phys(proxy) {sph:.4f}  cells {n}", flush=True)
import pandas as pd
R = pd.DataFrame(rows, columns=["panel", "world", "S_state", "S_FD", "S_LWR", "S_phys", "n"])
R["kind"] = np.where(R.world.str.startswith("T"), "T", R.world)
print(R.groupby("kind")[["S_state", "S_FD", "S_LWR", "S_phys"]].mean().round(4).to_string())
R.to_csv(MD / f"twin_eval_{tag}.csv", index=False)
