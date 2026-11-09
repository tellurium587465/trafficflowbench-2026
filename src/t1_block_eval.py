"""T1/T3 from the generator's blocks (10/14): adjacent links inside a block share one state (twin/block_structure.py:
split rate ~0, congested speed ratio = noise). A hidden target cell whose block-mates are visible in the same row can be
read from them. Blocks are estimated here WITHOUT February (split rate <= 0.05 over pairs congested >= 30 rows in the
other months), then February target cells are scored (held out from the T1 fit, as t1_snap_eval.py).
Variants (fixed a priori):
  A0 model f3big as is (production flags ADAPT/TRAINSTAT)
  A1 congested mates: if a visible block-mate in the row is congested, speed := median of the visible mates' speeds
  A2 A1 + flow := median of the mates' flows
  A3 any visible mates: speed := vf_target x median(v_mate / vf_mate), flow := median of the mates' flows
Reports S_state, the physics proxy, RMSE of speed in the touched cells and how many cells each variant touches.
usage: python t1_block_eval.py [panels=D7_I10_E,D12_I5_S]  (the panels t1_snap_eval read: likely still in the file cache)"""
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


def blocks_of(P, rows):
    V = P.V[rows]; ok = np.isfinite(V); cong = (V <= P.vcut[None, :]) & ok
    a, b = cong[:, :-1], cong[:, 1:]; both = ok[:, :-1] & ok[:, 1:]
    n_any = ((a | b) & both).sum(0); split = ((a ^ b) & both).sum(0) / np.maximum(n_any, 1)
    tight = (n_any >= 30) & (split <= 0.05)
    bid = np.zeros(P.n, int)
    for i in range(1, P.n):
        bid[i] = bid[i - 1] if tight[i - 1] else bid[i - 1] + 1
    return bid


def apply(P, Vp, Qp, r, c, bid, mode):
    if mode == "A0":
        return Vp, Qp, 0
    vis = np.isfinite(P.V) & ~P.target
    Vn, Qn = Vp.copy(), Qp.copy(); n = 0
    for i, j in zip(r, c):
        mates = np.flatnonzero((bid == bid[j]) & vis[i]); mates = mates[mates != j]
        if not len(mates):
            continue
        vm, qm = P.V[i, mates], P.Q[i, mates]
        if mode in ("A1", "A2"):
            cm = vm <= P.vcut[mates]
            if not cm.any():
                continue
            Vn[i, j] = np.median(vm[cm])
            if mode == "A2" and np.isfinite(qm[cm]).any():
                Qn[i, j] = np.nanmedian(qm[cm])
        else:
            Vn[i, j] = P.vf[j] * np.median(vm / P.vf[mates])
            if np.isfinite(qm).any():
                Qn[i, j] = np.nanmedian(qm)
        n += 1
    return Vn, Qn, n


for p in (sys.argv[1] if len(sys.argv) > 1 else "D7_I10_E,D12_I5_S").split(","):
    pi = PANELS.index(p); P = load(p, "train")
    feb = np.array([str(s).startswith("2031-02") for s in P.times])
    bid = blocks_of(P, ~feb); print(f"{p}: {P.n} links in {bid.max() + 1} blocks", flush=True)
    F, _, _, _ = gf(P); bv, bq = bf(F)
    r, c = np.nonzero(P.target & feb[:, None] & P.E & np.isfinite(P.V))
    X, _ = rf(P, F, r, c); X = np.column_stack([np.full(len(r), pi, np.float32), X])
    Vp = np.full((P.T, P.n), np.nan); Qp = Vp.copy()
    Vp[r, c] = np.clip((bv[r, c] + mv.predict(X, num_threads=NT)) * P.vf[c], 3.0, 1.15 * P.vf[c])
    Qp[r, c] = np.clip((bq[r, c] + mq.predict(X, num_threads=NT)) * P.lanes[c], 60.0, None)
    del F, X
    tg, rg = P.target.copy(), P.regime.copy()
    P.target = P.target & feb[:, None]; P.regime = np.where(feb, P.regime, 255).astype(np.uint8)
    for md in ("A0", "A1", "A2", "A3"):
        V2, Q2, n = apply(P, Vp, Qp, r, c, bid, md)
        s = state_score(P, V2, Q2); ph = physics_score(P, V2, Q2)
        tch = np.abs(V2[r, c] - Vp[r, c]) > 1e-9
        rm0 = np.sqrt(np.nanmean((Vp[r, c][tch] - P.V[r, c][tch]) ** 2)) if tch.any() else np.nan
        rm1 = np.sqrt(np.nanmean((V2[r, c][tch] - P.V[r, c][tch]) ** 2)) if tch.any() else np.nan
        print(f"{p} {md}: S_state {s:.5f}  S_FD {ph[0]:.4f}  S_LWR(proxy) {ph[1]:.4f}  S_phys {ph[2]:.4f} | touched {n} of {len(r)}, "
              f"speed RMSE there {rm0:.2f} -> {rm1:.2f}", flush=True)
    P.target, P.regime = tg, rg
    load.cache_clear()
