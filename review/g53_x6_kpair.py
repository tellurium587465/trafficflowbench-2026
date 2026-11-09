"""G53: X6 local proxy (STATE.md 10/22 pre-registration; ideas T1.26). Inside runs of >= 2 time-consecutive Task 1
target cells of one link whose cells just before and just after are public (observed in the masked layer: finite, not
a target, not a masked-layer blank), pull the density k = q / v toward the linear interpolation of the two public k
by alpha, keeping v (q = k v) or keeping q (v = q / k). Base = f3big Feb predictions (Feb is held out of f3big) + the
production FD blend (t1_fd.py, a = 0.75). Scores on Feb: S_state (score_task1 replica), S_FD and S_LWR proxy
(metrics.physics_score), total-equivalent 0.35 S_state + 0.15 S_phys; family / panel means as t1_post_eval.py.
usage: python g53_x6_kpair.py"""
import sys
sys.path.insert(0, r"C:/Users/user/tfb/src")
import numpy as np, lightgbm as lgb
from pathlib import Path
from cachelib import load
from common import PANELS
from t1_exp import featfns
from metrics import state_score, physics_score, family_mean

MD = Path(r"C:/Users/user/tfb/work/t1"); gf, rf, bf = featfns(3)
mv = lgb.Booster(model_file=str(MD / "speed_f3big.txt")); mq = lgb.Booster(model_file=str(MD / "flow_f3big.txt"))


def fd(P, V, Q, c, a=0.75):                                       # t1_fd.py
    cong = V < 0.6 * P.vf[c]; k = np.maximum(P.kj[c] - Q / P.w[c], 1e-3)
    vn = np.where(cong, (1 - a) * V + a * Q / k, V); return np.maximum(vn, Q / P.kj[c]), Q


def kpair(P, Vg, Qg, tgt, alpha, keep):
    """Vg, Qg: (T, n) grids with the predictions at tgt; returns new grids and the number of cells changed"""
    pub = np.isfinite(P.V) & np.isfinite(P.Q) & ~P.target & ~P.blank
    Vn, Qn = Vg.copy(), Qg.copy(); nch = 0
    for j in range(P.n):
        col = tgt[:, j]; i = 0; T = len(col)
        while i < T:
            if not col[i]:
                i += 1; continue
            e = i
            while e + 1 < T and col[e + 1]:
                e += 1
            if e > i and i > 0 and e + 1 < T and pub[i - 1, j] and pub[e + 1, j]:
                k0 = P.Q[i - 1, j] / max(P.V[i - 1, j], 1.0); k1 = P.Q[e + 1, j] / max(P.V[e + 1, j], 1.0)
                L = e - i + 2; ki = k0 + (k1 - k0) * np.arange(1, L) / L
                kp = Qg[i:e + 1, j] / np.maximum(Vg[i:e + 1, j], 1.0); kn = (1 - alpha) * kp + alpha * ki
                if keep == "v":
                    Qn[i:e + 1, j] = kn * np.maximum(Vg[i:e + 1, j], 1.0)
                else:
                    Vn[i:e + 1, j] = np.clip(Qg[i:e + 1, j] / np.maximum(kn, 1e-3), 3.0, 1.15 * P.vf[j])
                nch += e - i + 1
            i = e + 1
    return Vn, Qn, nch


VARS = [("base", None, None)] + [(f"a{a}_keep{k}", a, k) for a in (0.5, 1.0) for k in ("v", "q")]
tot = {nm: {} for nm, _, _ in VARS}
for pi, panel in enumerate(PANELS):
    P = load(panel, "train"); F, _, _, _ = gf(P); bv, bq = bf(F)
    feb = np.array([s.startswith("2031-02") for s in P.times]); tgt = P.target & feb[:, None]; r, c = np.nonzero(tgt)
    X, _ = rf(P, F, r, c); X = np.column_stack([np.full(len(r), pi, np.float32), X])
    V = (bv[r, c] + mv.predict(X, num_threads=16)) * P.vf[c]; Q = (bq[r, c] + mq.predict(X, num_threads=16)) * P.lanes[c]
    V, Q = fd(P, np.clip(V, 3.0, 1.15 * P.vf[c]), np.clip(Q, 60.0, None), c)
    Vg = np.full((P.T, P.n), np.nan); Qg = Vg.copy(); Vg[r, c] = V; Qg[r, c] = Q
    tg, rg = P.target.copy(), P.regime.copy(); P.target = tgt; P.regime = np.where(feb, P.regime, 255).astype(np.uint8)
    line = []
    for nm, a, k in VARS:
        Vn, Qn, nch = (Vg, Qg, 0) if a is None else kpair(P, Vg, Qg, tgt, a, k)
        s = state_score(P, Vn, Qn); ph = physics_score(P, Vn, Qn)
        tot[nm][panel] = (s, ph[0], ph[1], ph[2]); line.append(f"{nm} S {s:.4f} FD {ph[0]:.4f} LWR {ph[1]:.4f} ({nch} cells)")
    P.target, P.regime = tg, rg
    print(panel, f"targets {len(r)} |", " | ".join(line), flush=True)
base = {i: family_mean({p: v[i] for p, v in tot["base"].items()}) for i in range(4)}
print(f"base: S_state {base[0]:.5f} S_FD {base[1]:.5f} S_LWR_proxy {base[2]:.5f} S_phys {base[3]:.5f}")
for nm, a, k in VARS[1:]:
    m = {i: family_mean({p: v[i] for p, v in tot[nm].items()}) for i in range(4)}
    d = {i: m[i] - base[i] for i in range(4)}; te = 0.35 * d[0] + 0.15 * d[3]
    ok = d[0] >= -0.0005 and d[2] > 0
    print(f"{nm}: dS_state {d[0]:+.5f} dS_FD {d[1]:+.5f} dS_LWR {d[2]:+.5f} dS_phys {d[3]:+.5f} | total-equivalent {te:+.5f} | "
          f"panels with dS_LWR > 0: {sum(tot[nm][p][2] > tot['base'][p][2] for p in tot[nm])}/{len(tot[nm])} | {'meets the condition' if ok else 'fails'}")
