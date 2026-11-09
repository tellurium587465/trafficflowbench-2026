"""T1/T3 idea from the generator study (IDEAS_TOP3, 10/14): real congested speeds sit at site-specific discrete levels
inside an episode (spread 0.01-0.18 of vf). For a hidden target cell whose visible neighbours (same link rows t-2..t+2,
links c-1..c+1 at rows t-1..t+1) are congested and whose model prediction is congested too, pull the predicted speed
toward the neighbours' median: v' = (1-a) v + a * med. Flow is moved along the same density (q' = q * v'/v keeps
k = q/v), so only the speed changes the FD point. a in {0, 0.5, 1} (fixed a priori). Model f3big (feature version 3,
production flags ADAPT/TRAINSTAT), February target cells (held out from the fit), two panels only (small reads).
Reports S_state and the physics proxy (metrics.physics_score) per a.
usage: python t1_snap_eval.py [panels=D12_I5_S,D7_I10_E]"""
import os, sys
os.environ.setdefault("TFB_T1_ADAPT", "1"); os.environ.setdefault("TFB_T1_FEAT_TRAINSTAT", "1")
import numpy as np, lightgbm as lgb
from pathlib import Path
from cachelib import load
from common import PANELS
from t1_exp import featfns
from metrics import state_score, physics_score

MD = Path(r"C:/Users/user/tfb/work/t1"); NT = int(os.environ.get("TFB_THREADS", "8"))
gf, rf, bf = featfns(3)
mv = lgb.Booster(model_file=str(MD / "speed_f3big.txt")); mq = lgb.Booster(model_file=str(MD / "flow_f3big.txt"))


def snap(P, Vp, Qp, r, c, a):
    if a == 0:
        return Vp, Qp
    V = np.where(P.target & ~np.isnan(P.V), np.nan, P.V)         # what the model may see: non-target observations
    cong = V <= P.vcut[None, :]
    Vn, Qn = Vp.copy(), Qp.copy(); n = 0
    for i, j in zip(r, c):
        if not Vp[i, j] <= P.vcut[j]:
            continue
        rs = slice(max(0, i - 2), i + 3); ls = slice(max(0, j - 1), j + 2)
        blk_v = np.concatenate([V[rs, j], V[i, ls], V[max(0, i - 1):i + 2, ls].ravel()])
        blk_c = np.concatenate([cong[rs, j], cong[i, ls], cong[max(0, i - 1):i + 2, ls].ravel()])
        ok = np.isfinite(blk_v)
        if ok.sum() >= 3 and blk_c[ok].mean() >= 0.8:
            med = np.median(blk_v[ok & blk_c]); vnew = (1 - a) * Vp[i, j] + a * med
            Qn[i, j] = Qp[i, j] * vnew / max(Vp[i, j], 1e-3); Vn[i, j] = vnew; n += 1
    print(f"   snapped cells {n}", flush=True)
    return Vn, Qn


for p in (sys.argv[1] if len(sys.argv) > 1 else "D12_I5_S,D7_I10_E").split(","):
    pi = PANELS.index(p); P = load(p, "train")
    F, _, _, _ = gf(P); bv, bq = bf(F)
    feb = np.array([str(s).startswith("2031-02") for s in P.times])
    r, c = np.nonzero(P.target & feb[:, None] & P.E & np.isfinite(P.V))
    X, _ = rf(P, F, r, c); X = np.column_stack([np.full(len(r), pi, np.float32), X])
    Vp = np.full((P.T, P.n), np.nan); Qp = Vp.copy()
    Vp[r, c] = np.clip((bv[r, c] + mv.predict(X, num_threads=NT)) * P.vf[c], 3.0, 1.15 * P.vf[c])
    Qp[r, c] = np.clip((bq[r, c] + mq.predict(X, num_threads=NT)) * P.lanes[c], 60.0, None)
    tg, rg = P.target.copy(), P.regime.copy()
    P.target = P.target & feb[:, None]; P.regime = np.where(feb, P.regime, 255).astype(np.uint8)
    for a in (0.0, 0.5, 1.0):
        V2, Q2 = snap(P, Vp, Qp, r, c, a)
        s = state_score(P, V2, Q2); ph = physics_score(P, V2, Q2)
        cg = (P.V[r, c] <= P.vcut[c]); err = np.abs(V2[r, c] - P.V[r, c])
        print(f"{p} a={a}: S_state {s:.5f}  S_FD {ph[0]:.4f}  S_LWR(proxy) {ph[1]:.4f}  S_phys {ph[2]:.4f} | congested target cells {int(cg.sum())}, speed MAE there {np.nanmean(err[cg]):.2f}", flush=True)
    P.target, P.regime = tg, rg
    del F; load.cache_clear()
