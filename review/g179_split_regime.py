"""G179 (11/01, ledger #130, reopening #40): cells where the models disagree on the regime. Task 3 is an L1 on N, so on a two-peaked cell the median of the models may beat their mean. On the Feb harness through the production chain, with the #118b mix (0.6 / 0.1):
split cell = target cell where the chain, the nets of seeds 0, 1 and the nets of seeds 2, 3 are not on the same side of 0.6 vf. On split cells the mix is replaced by the median of the three, speed and flow. Same read as G177.
usage: py -B -X utf8 g179_split_regime.py"""
import os, sys, pickle, copy
src = open(r"C:/Users/user/tfb/audit/b29_stage2_eval.py", encoding="utf-8").read().split('D = pd.read_parquet(WORK + "/t1/b28_rows.parquet")')[0]
sys.argv = sys.argv[:1]; sys.dont_write_bytecode = True; exec(compile(src, "b29", "exec"), globals())
UH = "C:/tfb_data/tfb_nn/unet_h/"; ST = {p: stage1(p) for p in PANELS}; R = {}


def clip(P, S, V, Q):
    return np.maximum(np.minimum(V, S["cap"][None, :]), Q / P.kj[None, :])


for p in PANELS:
    S = ST[p]; P, Pg = harness(p, S); val = P.ramp_validity(); V0, Q0 = S["Vb"], S["Qp"]; Vf, Qf = post(P, S, V0, Q0); gap = S["gap"]; day = np.arange(P.T) // 288; tv = Pg.target & np.isfinite(P.V) & np.isfinite(P.Q) & P.E & np.isfinite(V0)
    a = np.load(UH + f"{p}_unet.npz"); b = np.load(UH + f"{p}_unet_s23.npz"); Va, Vb_ = np.clip(a["VU"].astype(float), 3, 1.15 * P.vf[None, :]), np.clip(b["VU"].astype(float), 3, 1.15 * P.vf[None, :]); Qa, Qb = np.clip(a["QU"].astype(float), 60, None), np.clip(b["QU"].astype(float), 60, None)
    w = np.where(gap, 0.1, 0.6); Vm = np.where(tv, (1 - w) * Vf + w * Vb_, Vf); Qm = np.where(tv, (1 - w) * Qf + w * Qb, Qf); cut = 0.6 * P.vf[None, :]; c = (Vf < cut).astype(int) + (Va < cut).astype(int) + (Vb_ < cut).astype(int); sp = tv & (c > 0) & (c < 3)
    Vd = np.median(np.stack([Vf, Va, Vb_]), axis=0); Qd = np.median(np.stack([Qf, Qa, Qb]), axis=0); V2 = np.where(sp, Vd, Vm); Q2 = np.where(sp, Qd, Qm); R[p] = dict(share=float(sp.sum() / tv.sum()))
    for par in (0, 1):
        Pe = copy.copy(Pg); Pe.target = Pg.target & ((day % 2 == par)[:, None]); x = (state_score(Pe, clip(P, S, Vm, Qm), Qm), physics_score(Pe, clip(P, S, Vm, Qm), Qm, validity=val)[2]); y = (state_score(Pe, clip(P, S, V2, Q2), Q2), physics_score(Pe, clip(P, S, V2, Q2), Q2, validity=val)[2]); R[p][par] = (y[0] - x[0], y[1] - x[1])
    print(p, f"split share {R[p]['share']:.4f}", [round(0.35 * R[p][q][0] + 0.15 * R[p][q][1], 5) for q in (0, 1)], flush=True); load.cache_clear()
for par in (0, 1):
    v = np.array([0.35 * R[p][par][0] + 0.15 * R[p][par][1] for p in PANELS]); print(f"{'odd' if par == 0 else 'even'} days: {v.mean():+.5f} (SE {v.std(ddof=1) / np.sqrt(len(v)):.5f}) dS_state {np.mean([R[p][par][0] for p in PANELS]):+.5f} dS_phys {np.mean([R[p][par][1] for p in PANELS]):+.5f} panels > 0: {(v > 0).sum()}/10")
