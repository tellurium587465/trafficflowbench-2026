"""G191 (11/01, ledger #149, strategy T2_DEEP_REVIEW): Task 1 inside the T2 blanks as a two-point boundary problem. Task 1 is offline, so the state after a blank is known. CTM (t2_ctm.cells, released FD) from the row before the blank, grid over the head of the queue (the most downstream congested cell at either end), its capacity factor c_r until the clearing time t_e, and the inflow scale; the runs closest to the speeds after the blank are kept: best, or the mean of the 5 closest. Link level: the run + the link's offset at both ends, linear in time. Blanks with no congested cell at either end are left alone.
Feb harness of B29, reference = chain + net _s23 at 0.6 / 0.1. Arms = (best | top5) x w. Gate, then the line: 0.35 dS_state + 0.15 dS_phys, chosen on odd days, read on even days.
usage: py -B -X utf8 g191_blank_ctm.py"""
import os, sys, pickle, copy, time
src = open(r"C:/Users/user/tfb/audit/b29_stage2_eval.py", encoding="utf-8").read().split('D = pd.read_parquet(WORK + "/t1/b28_rows.parquet")')[0]
sys.argv = sys.argv[:1]; sys.dont_write_bytecode = True; exec(compile(src, "b29", "exec"), globals())
from t2_ctm import cells
UH = "C:/tfb_data/tfb_nn/unet_h/"; ST = {p: stage1(p) for p in PANELS}; print("first stage ready", flush=True); T0 = time.time()
CR = (0.5, 0.6, 0.7, 0.8, 0.9, 1.0); TE = (0.0, 0.25, 0.5, 0.75, 1.0, 1.5); AL = (0.8, 0.9, 1.0, 1.1, 1.2); WS = (0.1, 0.2, 0.3, 0.5); NR = 19


def clip(P, S, V, Q):
    return np.maximum(np.minimum(V, S["cap"][None, :]), Q / P.kj[None, :])


def sim(P, k0, q0, q1, heads):
    g, nc, Lc, vf, cap, kj, kc, w, dt = cells(P); A = np.array([(h, c, t_, a) for h in heads for c in CR for t_ in TE for a in AL]); H = A[:, 0].astype(int); m = len(A); idx = np.arange(m); ks = np.tile(k0, (m, 1)); sp = int(round((5 / 60) / dt)); dt_ = (5 / 60) / sp; rec = np.zeros((NR, m, nc), np.float32); n_ = sp * NR
    for s in range(1, n_ + 1):
        tau = s / n_; Sd = np.minimum(vf * ks, cap); Sd[idx, H] = np.where(tau < A[:, 2], np.minimum(Sd[idx, H], cap[H] * A[:, 1]), Sd[idx, H]); Rc = np.minimum(cap, w * (kj - ks)); f = np.minimum(Sd[:, :-1], Rc[:, 1:]); fin = np.minimum(A[:, 3] * ((1 - tau) * q0 + tau * q1), Rc[:, 0]); dk = np.empty_like(ks); dk[:, 0] = fin - f[:, 0]; dk[:, 1:-1] = f[:, :-1] - f[:, 1:]; dk[:, -1] = f[:, -1] - Sd[:, -1]; ks = np.clip(ks + dt_ * dk / Lc, 0, kj)
        if s % sp == 0:
            rec[s // sp - 1] = ks
    return rec


def fl(x):
    ok = np.isfinite(x); i_ = np.arange(len(x)); return np.interp(i_, i_[ok], x[ok])


def speed(P, k):
    g, nc, Lc, vf, cap, kj, kc, w, dt = cells(P); return np.minimum(vf, w * (kj - k) / np.maximum(k, 1e-6)).clip(1.0, None)


def flow(P, k):
    g, nc, Lc, vf, cap, kj, kc, w, dt = cells(P); return np.minimum(vf * k, cap)


R = {}; GT = []
for p in PANELS:
    S = ST[p]; P, Pg = harness(p, S); val = P.ramp_validity(); V0, Q0 = S["Vb"], S["Qp"]; Vf, Qf = post(P, S, V0, Q0); gap = np.broadcast_to(S["gap"], P.V.shape) if np.ndim(S["gap"]) == 2 else np.broadcast_to(np.asarray(S["gap"])[:, None], P.V.shape); day = np.arange(P.T) // 288; tv = Pg.target & np.isfinite(P.V) & np.isfinite(P.Q) & P.E & np.isfinite(V0); R[p] = {}
    u = np.load(UH + f"{p}_unet_s23.npz"); w_ = np.where(gap, 0.1, 0.6); Qb = np.where(tv, (1 - w_) * Qf + w_ * np.clip(u["QU"].astype(float), 60.0, None), Qf); Vb = clip(P, S, np.where(tv, (1 - w_) * Vf + w_ * np.clip(u["VU"].astype(float), 3.0, 1.15 * P.vf[None, :]), Vf), Qb); vis = np.isfinite(P.V) & np.isfinite(P.Q) & ~Pg.blank & ~Pg.target; Ev = np.where(vis, P.V, Vb); Eq = np.where(vis, P.Q, Qb); g, nc, Lc, vf, cap, kj, kc, w, dt = cells(P); Ls = P.length.clip(1e-3)
    PE = []
    for par in (0, 1):
        Pe = copy.copy(Pg); Pe.target = Pg.target & ((day % 2 == par)[:, None]); PE.append((Pe, state_score(Pe, Vb, Qb), physics_score(Pe, Vb, Qb, validity=val)[2]))
    VC = {"best": Vb.copy(), "top5": Vb.copy()}; QC = {"best": Qb.copy(), "top5": Qb.copy()}; nb = 0
    for t, _ in S["wl"]:
        r0, r1 = t, t + NR; v_0, v_1, q_0, q_1 = fl(Ev[r0]), fl(Ev[r1]), fl(Eq[r0]), fl(Eq[r1]); k_l = np.where(v_0 < 0.6 * P.vf, P.kj * P.w / (v_0 + P.w), q_0 / np.maximum(v_0, 1.0)) if os.environ.get("G191_K", "") == "fd" else q_0 / np.maximum(v_0, 1.0); pre = np.bincount(g, k_l * Ls, nc) / Lc; aft = np.bincount(g, v_1 * Ls, nc) / Lc; vp = np.bincount(g, v_0 * Ls, nc) / Lc; cp = vp < 0.6 * vf; ca = aft < 0.6 * vf
        if not (cp.any() or ca.any()):
            continue
        heads = sorted({int(np.flatnonzero(x)[-1]) for x in (cp, ca) if x.any()}); rec = sim(P, np.clip(pre, 0, kj), float(q_0[:3].mean()), float(q_1[:3].mean()), heads); ve = speed(P, rec[-1]); d_ = (((ve - aft) ** 2) * Lc).sum(axis=1); o = np.argsort(d_); nb += 1
        for nm, top in (("best", o[:1]), ("top5", o[:5])):
            vs = speed(P, rec[:, top]).mean(axis=1); qs = flow(P, rec[:, top]).mean(axis=1); dv0 = v_0 - speed(P, np.clip(pre, 0, kj))[g]; dv1 = v_1 - vs[-1][g]; dq0 = q_0 - flow(P, np.clip(pre, 0, kj))[g]; dq1 = q_1 - qs[-1][g]
            for j in range(NR - 1):
                tau = (j + 1) / NR; rr = r0 + 1 + j; VC[nm][rr] = np.clip(vs[j][g] + (1 - tau) * dv0 + tau * dv1, 3.0, 1.15 * P.vf); QC[nm][rr] = np.maximum(qs[j][g] + (1 - tau) * dq0 + tau * dq1, 60.0)
        m_ = tv[r0 + 1:r1] & (P.V[r0 + 1:r1] < 0.6 * P.vf[None, :])
        if m_.sum() >= 20:
            T_ = P.V[r0 + 1:r1][m_]; tau_ = (np.arange(1, NR) / NR)[:, None]; lin = (1 - tau_) * v_0[None, :] + tau_ * v_1[None, :]; GT.append(dict(panel=p, t=t, n=int(m_.sum()), lin=float(np.sqrt(np.mean((lin[m_] - T_) ** 2))), end=float(np.sqrt(np.mean((speed(P, rec[-1, o[:1]])[0] - aft) ** 2))), start=float(np.sqrt(np.mean((speed(P, np.clip(pre, 0, kj)) - vp) ** 2))), base=float(np.sqrt(np.mean((Vb[r0 + 1:r1][m_] - T_) ** 2))), best=float(np.sqrt(np.mean((VC["best"][r0 + 1:r1][m_] - T_) ** 2))), top5=float(np.sqrt(np.mean((VC["top5"][r0 + 1:r1][m_] - T_) ** 2)))))
    for nm in VC:
        for w2 in WS:
            Qm = np.where(gap & tv, (1 - w2) * Qb + w2 * QC[nm], Qb); Vm = clip(P, S, np.where(gap & tv, (1 - w2) * Vb + w2 * VC[nm], Vb), Qm)
            for par, (Pe, a0, b0) in enumerate(PE):
                R[p][(par, nm, w2)] = (state_score(Pe, Vm, Qm) - a0, physics_score(Pe, Vm, Qm, validity=val)[2] - b0)
    print(p, "blanks run", nb, f"{time.time() - T0:.0f}s", flush=True); load.cache_clear()
G_ = pd.DataFrame(GT); G_.to_csv(WORK + "/t1/g191_gate.csv", index=False); print("blanks with 20+ congested target cells", len(G_), "| RMSE v base", round(float(np.sqrt((G_.base ** 2 * G_.n).sum() / G_.n.sum())), 3), "best", round(float(np.sqrt((G_.best ** 2 * G_.n).sum() / G_.n.sum())), 3), "top5", round(float(np.sqrt((G_.top5 ** 2 * G_.n).sum() / G_.n.sum())), 3), "| share of blanks <= base: best", round(float((G_.best <= G_.base).mean()), 3), "top5", round(float((G_.top5 <= G_.base).mean()), 3), "(gate 0.3) | control: linear in time RMSE", round(float(np.sqrt((G_.lin ** 2 * G_.n).sum() / G_.n.sum())), 3), "| cell speed misfit at the start", round(float(G_.start.mean()), 2), "at the end of the best run", round(float(G_.end.mean()), 2))
PN = list(PANELS); obj = lambda par, key: np.array([0.35 * R[p][(par, *key)][0] + 0.15 * R[p][(par, *key)][1] for p in PN]); KEYS = sorted({k[1:] for k in R[PN[0]]}, key=str)
for sel, rd, nm in ((0, 1, "chosen on odd days, read on even days (the line)"), (1, 0, "reverse (record)")):
    b = max(KEYS, key=lambda c: obj(sel, c).mean()); v = obj(rd, b); print(f"{nm}: {b} | selection {obj(sel, b).mean():+.5f} | read {v.mean():+.5f} (SE {v.std(ddof=1) / np.sqrt(len(v)):.5f}) | dS_state {np.mean([R[p][(rd, *b)][0] for p in PN]):+.5f} dS_phys {np.mean([R[p][(rd, *b)][1] for p in PN]):+.5f} | panels > 0: {(v > 0).sum()}/10", flush=True)
for k in KEYS:
    print(f"  {k}: odd {obj(0, k).mean():+.5f} even {obj(1, k).mean():+.5f}")
