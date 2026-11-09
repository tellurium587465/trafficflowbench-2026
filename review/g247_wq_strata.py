"""G247 (11/04, ledger #211b, the next stage of #211): the net's weight on the flow channel only, by corridor x day (06-21 h) / night,
outside the T2 blanks (inside them it stays 0.1; the speed channel stays 0.7 / 0.1). G245's February harness, no retraining.
w_q from {0.4 .. 1.0} chosen per stratum on odd calendar days (minimum flow SSE per lane on the hidden cells), read on even days by
the Task 1 state score (metrics.state_score, the density clip v >= q / k_j as G168) against the production mix, per corridor.
Line (#211b): even days, mean dS_T1 >= +0.0001 and >= 7/10 corridors non-negative. usage: py -B -X utf8 g247_wq_strata.py"""
import copy, numpy as np
R_ = r"C:/Users/user/tfb/review/"
s5 = open(R_ + "g245_t1_flow_postfix.py", encoding="utf-8").read().split("XS, YS, DS, BS = [], [], [], []")[0]; g5 = {"__name__": "g245"}; exec(compile(s5, "g245b", "exec"), g5)
from metrics import state_score, physics_score
WG = (0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0); out = []
for p in g5["PANELS"]:
    P0 = g5["load"](p, "train"); P, Pg, gap, wl, a0 = g5["view"](P0); z = np.load(g5["O"] + p + ".npz"); u = np.load(g5["O"] + p + "_unet_s45.npz")
    V0, Q0 = z["V0"].astype(float), z["Q0"].astype(float); gap = z["gap"]; VU = np.clip(u["VU"].astype(float), 3.0, 1.15 * P.vf[None, :]); QU = np.clip(u["QU"].astype(float), 60.0, None)
    tv = Pg.target & np.isfinite(P.V) & np.isfinite(P.Q) & P.E & np.isfinite(V0); day = np.arange(P.T) // 288; hr = (np.arange(P.T) % 288) // 12; dm = ((hr >= 6) & (hr < 21))[:, None]
    wv = np.where(gap, 0.1, 0.7); Vm = np.where(tv, (1 - wv) * V0 + wv * VU, V0); Qp = np.where(tv, (1 - wv) * Q0 + wv * QU, Q0)
    sel = tv & ~gap & ((day % 2 == 0)[:, None]); wq = np.where(gap, 0.1, 0.7); ch = {}
    for per, m in (("day", dm), ("night", ~dm)):
        mm = sel & m; sse = {w: float(np.sum((((1 - w) * Q0 + w * QU - P.Q) / P.lanes[None, :])[mm] ** 2)) for w in WG}; b = min(WG, key=sse.get); ch[per] = b
        wq = np.where(~gap & np.broadcast_to(m, gap.shape), b, wq)
    Qn = np.where(tv, (1 - wq) * Q0 + wq * QU, Q0); Pe = copy.copy(Pg); Pe.target = Pg.target & ((day % 2 == 1)[:, None]); val = P.ramp_validity()
    s0 = state_score(Pe, np.maximum(Vm, Qp / P.kj[None, :]), Qp); s1 = state_score(Pe, np.maximum(Vm, Qn / P.kj[None, :]), Qn)
    f0 = physics_score(Pe, np.maximum(Vm, Qp / P.kj[None, :]), Qp, validity=val)[2]; f1 = physics_score(Pe, np.maximum(Vm, Qn / P.kj[None, :]), Qn, validity=val)[2]
    out.append((p, ch["day"], ch["night"], s1 - s0, f1 - f0)); print(f"{p}: w_q day {ch['day']} night {ch['night']} | even days dS_T1 {s1 - s0:+.6f} dS_phys {f1 - f0:+.6f}", flush=True); g5["load"].cache_clear()
d = np.array([o[3] for o in out]); f = np.array([o[4] for o in out])
print(f"mean dS_T1 {d.mean():+.6f} (SE {d.std(ddof=1) / np.sqrt(len(d)):.6f}), corridors >= 0: {(d >= 0).sum()}/10 | dS_phys {f.mean():+.6f} | S (0.35 T1 + 0.15 T3 phys) {0.35 * d.mean() + 0.15 * f.mean():+.6f} | line (>= +0.0001 and >= 7/10): {d.mean() >= 1e-4 and (d >= 0).sum() >= 7}")
