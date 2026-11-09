"""G186 (11/01, ledger #140, strategy T2_DEEP_REVIEW 18): where the S_LWR residual sits and two post-processes on N = q L / v. Feb harness through the production chain with the #118b mix (0.6 / 0.1). Residual mass |dN_pred - dN_true| over rows, by cell kind: scattered blank or T2 blank, free or congested, boundary.
  med : inside the T2 blanks, N is median-filtered along time (5 rows), v kept, q = N v / L
  rom : on target cells predicted congested, N = the mix of the two N (chain, net) in place of the N of the mixed q and v; v kept, q = N v / L
Read as G177 against the fixed mix. Line: even days 0.35 dT1 + 0.15 dT3 >= +0.0002, 7 of 10 panels -> LB pair.
usage: py -B -X utf8 g186_t3_residual.py"""
import os, sys, pickle, copy
src = open(r"C:/Users/user/tfb/audit/b29_stage2_eval.py", encoding="utf-8").read().split('D = pd.read_parquet(WORK + "/t1/b28_rows.parquet")')[0]
sys.argv = sys.argv[:1]; sys.dont_write_bytecode = True; exec(compile(src, "b29", "exec"), globals())
from scipy.ndimage import median_filter
UH = "C:/tfb_data/tfb_nn/unet_h/"; ST = {p: stage1(p) for p in PANELS}; R = {}; MASS = {}


def clip(P, S, V, Q):
    return np.maximum(np.minimum(V, S["cap"][None, :]), Q / P.kj[None, :])


for p in PANELS:
    S = ST[p]; P, Pg = harness(p, S); val = P.ramp_validity(); V0, Q0 = S["Vb"], S["Qp"]; Vf, Qf = post(P, S, V0, Q0); gap = S["gap"]; day = np.arange(P.T) // 288; tv = Pg.target & np.isfinite(P.V) & np.isfinite(P.Q) & P.E & np.isfinite(V0)
    u = np.load(UH + f"{p}_unet_s23.npz"); VU = np.clip(u["VU"].astype(float), 3.0, 1.15 * P.vf[None, :]); QU = np.clip(u["QU"].astype(float), 60.0, None); w = np.where(gap, 0.1, 0.6); Vm = np.where(tv, (1 - w) * Vf + w * VU, Vf); Qm = np.where(tv, (1 - w) * Qf + w * QU, Qf); Vm = clip(P, S, Vm, Qm); L = P.length[None, :]
    N = Qm * L / np.maximum(Vm, 1.0); Nf = Qf * L / np.maximum(Vf, 1.0); Nu = QU * L / np.maximum(VU, 1.0); Nt = P.Q * L / np.maximum(P.V, 1.0); N_ = np.where(tv, N, Nt); ok = P.E & np.isfinite(P.V) & np.isfinite(P.Q); tr = ok[1:] & ok[:-1] & (tv[1:] | tv[:-1]); res = np.abs((N_[1:] - N_[:-1]) - (Nt[1:] - Nt[:-1])); cong = (P.V < 0.6 * P.vf[None, :]); b_ = np.zeros(gap.shape, bool); b_[S["rb"], S["cb"]] = True; kind = np.where(gap[1:], 2, 0) + cong[1:].astype(int) + np.where(b_[1:] & ~gap[1:], 4, 0)
    MASS[p] = {int(k): float(res[tr & (kind == k)].sum()) for k in np.unique(kind[tr])}; MASS[p]["cells"] = {int(k): int((tr & (kind == k)).sum()) for k in np.unique(kind[tr])}
    Nm = median_filter(np.nan_to_num(N_), size=(5, 1), mode="nearest"); Qa = np.where(tv & gap, Nm * np.maximum(Vm, 1.0) / L, Qm); cn = tv & (Vm < 0.6 * P.vf[None, :]); Nr = (1 - w) * Nf + w * Nu; Qb = np.where(cn, Nr * np.maximum(Vm, 1.0) / L, Qm); R[p] = {}
    for par in (0, 1):
        Pe = copy.copy(Pg); Pe.target = Pg.target & ((day % 2 == par)[:, None]); a0, b0 = state_score(Pe, Vm, Qm), physics_score(Pe, Vm, Qm, validity=val)[2]
        for nm, Qx in (("med", Qa), ("rom", Qb)):
            Vx = clip(P, S, Vm, Qx); R[p][(par, nm)] = (state_score(Pe, Vx, Qx) - a0, physics_score(Pe, Vx, Qx, validity=val)[2] - b0)
    print(p, flush=True); load.cache_clear()
KN = {0: "scattered free", 1: "scattered congested", 2: "T2 blank free", 3: "T2 blank congested", 4: "boundary free", 5: "boundary congested"}; tot = {k: sum(MASS[p].get(k, 0.0) for p in PANELS) for k in KN}; cel = {k: sum(MASS[p]["cells"].get(k, 0) for p in PANELS) for k in KN}; s_ = sum(tot.values())
for k in KN:
    print(f"{KN[k]:22s} mass {tot[k] / s_:.3f} cells {cel[k] / sum(cel.values()):.3f} per cell {tot[k] / max(cel[k], 1):.3f}")
for nm in ("med", "rom"):
    for par in (0, 1):
        v = np.array([0.35 * R[p][(par, nm)][0] + 0.15 * R[p][(par, nm)][1] for p in PANELS]); print(f"{nm} {'odd' if par == 0 else 'even'}: {v.mean():+.5f} (SE {v.std(ddof=1) / np.sqrt(len(v)):.5f}) dS_state {np.mean([R[p][(par, nm)][0] for p in PANELS]):+.5f} dS_phys {np.mean([R[p][(par, nm)][1] for p in PANELS]):+.5f} panels > 0: {(v > 0).sum()}/10", flush=True)
