"""G188 (11/01, ledger #147, strategy): is there a corridor-wide row component in the flow noise, as in the speed noise (+0.33, #57)? Feb harness through the chain with the #118b mix. (1) Correlation of the flow residual (mix - truth) / lanes between target cells of one row at distance 1, 2, 3 links, in free flow. (2) The correction as #57: c_t = the median over the visible free cells of the row of the flow against the median of the same link's visible rows t-3 .. t+3, target free cells get the mix + g c_t, g by the slope on the other panels. Read as G177.
Line: rho >= 0.1 and even days 0.35 dT1 + 0.15 dT3 >= +0.0002, 7 of 10 -> LB pair. rho < 0.1 -> Task 3 closes at the flow floor of Task 1.
usage: py -B -X utf8 g188_flow_shock.py"""
import os, sys, pickle, copy
src = open(r"C:/Users/user/tfb/audit/b29_stage2_eval.py", encoding="utf-8").read().split('D = pd.read_parquet(WORK + "/t1/b28_rows.parquet")')[0]
sys.argv = sys.argv[:1]; sys.dont_write_bytecode = True; exec(compile(src, "b29", "exec"), globals())
from scipy.ndimage import median_filter
UH = "C:/tfb_data/tfb_nn/unet_h/"; ST = {p: stage1(p) for p in PANELS}; RHO = {}; PA = {}; R = {}


def clip(P, S, V, Q):
    return np.maximum(np.minimum(V, S["cap"][None, :]), Q / P.kj[None, :])


for p in PANELS:
    S = ST[p]; P, Pg = harness(p, S); V0, Q0 = S["Vb"], S["Qp"]; Vf, Qf = post(P, S, V0, Q0); gap = S["gap"]; tv = Pg.target & np.isfinite(P.V) & np.isfinite(P.Q) & P.E & np.isfinite(V0); u = np.load(UH + f"{p}_unet_s23.npz"); w = np.where(gap, 0.1, 0.6); Vm = np.where(tv, (1 - w) * Vf + w * np.clip(u["VU"].astype(float), 3, 1.15 * P.vf[None, :]), Vf); Qm = np.where(tv, (1 - w) * Qf + w * np.clip(u["QU"].astype(float), 60, None), Qf); Vm = clip(P, S, Vm, Qm)
    fr = tv & (Vm >= 0.85 * P.vf[None, :]) & ~gap; e = np.where(fr, (Qm - P.Q) / P.lanes[None, :], np.nan); rr = []
    for d_ in (1, 2, 3):
        a, b = e[:, :-d_], e[:, d_:]; ok = np.isfinite(a) & np.isfinite(b); rr.append(float(np.corrcoef(a[ok], b[ok])[0, 1]) if ok.sum() > 100 else np.nan)
    vis = np.isfinite(P.V) & np.isfinite(P.Q) & ~Pg.target & ~Pg.blank & (P.V >= 0.85 * P.vf[None, :]); ql = np.where(vis, P.Q / P.lanes[None, :], np.nan); base = np.nanmedian(np.stack([np.roll(ql, s_, axis=0) for s_ in (-3, -2, -1, 1, 2, 3)]), axis=0); dev_ = np.clip(ql - base, -300, 300); nv = np.isfinite(dev_).sum(axis=1); c_ = np.where(nv >= 20, np.nanmedian(np.where(np.isfinite(dev_), dev_, np.nan), axis=1), 0.0); ok = np.isfinite(e); x = np.broadcast_to(c_[:, None], e.shape)[ok]; slope = float((x * -e[ok]).sum() / max((x * x).sum(), 1e-9))
    RHO[p] = rr; PA[p] = (float((x * x).sum()), float((x * -e[ok]).sum())); R[p] = dict(c=c_, fr=fr, Vm=Vm, Qm=Qm, Pg=Pg, P=P, S=S); print(p, "rho", [round(x_, 3) for x_ in rr], "slope", round(slope, 3), flush=True); load.cache_clear()
print("mean rho", np.round(np.nanmean(np.array(list(RHO.values())), axis=0), 3)); out = {}
for p in PANELS:
    d = R[p]; P, Pg, S = d["P"], d["Pg"], d["S"]; g_ = sum(PA[q][1] for q in PANELS if q != p) / max(sum(PA[q][0] for q in PANELS if q != p), 1e-9); Q2 = np.where(d["fr"], d["Qm"] + g_ * d["c"][:, None] * P.lanes[None, :], d["Qm"]); val = P.ramp_validity(); day = np.arange(P.T) // 288; V2 = clip(P, S, d["Vm"], Q2)
    for par in (0, 1):
        Pe = copy.copy(Pg); Pe.target = Pg.target & ((day % 2 == par)[:, None]); a = (state_score(Pe, d["Vm"], d["Qm"]), physics_score(Pe, d["Vm"], d["Qm"], validity=val)[2]); b = (state_score(Pe, V2, Q2), physics_score(Pe, V2, Q2, validity=val)[2]); out[(p, par)] = (b[0] - a[0], b[1] - a[1])
for par in (0, 1):
    v = np.array([0.35 * out[(p, par)][0] + 0.15 * out[(p, par)][1] for p in PANELS]); print(f"{'odd' if par == 0 else 'even'}: {v.mean():+.5f} (SE {v.std(ddof=1) / np.sqrt(len(v)):.5f}) dT1 {np.mean([out[(p, par)][0] for p in PANELS]):+.5f} dT3 {np.mean([out[(p, par)][1] for p in PANELS]):+.5f} panels > 0 {(v > 0).sum()}/10")
