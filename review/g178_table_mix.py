"""G178 (11/01, ledger #126; strategy T2_DEEP_REVIEW 13): the mix weight of the image network by a table: panel x mask regime x {free, predicted congested, boundary cell, blank depth 1-3, 4-15, 16-18}, speed and flow apart, convex. On the Feb harness through the production chain:
w = sum (net - chain)(truth - chain) / sum (net - chain)^2 on the fitting half, cut to [0, 1], read on the other half against the fixed mix (0.6 outside, 0.1 inside the blanks, the #118b pair). Line: 0.35 dT1 + 0.15 dT3 >= +0.0001 against the fixed mix on the read half, 7 of 10 panels, the reverse with the same sign.
Also prints the #127 arms from g177_eval.pkl.
usage: py -B -X utf8 g178_table_mix.py [set=_s23]"""
import os, sys, pickle, copy
SET = (sys.argv[1:] or ["_s23"])[0]
src = open(r"C:/Users/user/tfb/audit/b29_stage2_eval.py", encoding="utf-8").read().split('D = pd.read_parquet(WORK + "/t1/b28_rows.parquet")')[0]
sys.argv = sys.argv[:1]; sys.dont_write_bytecode = True; exec(compile(src, "b29", "exec"), globals())
UH = "C:/tfb_data/tfb_nn/unet_h/"; ST = {p: stage1(p) for p in PANELS}; R = {}; WT = {}


def clip(P, S, V, Q):
    return np.maximum(np.minimum(V, S["cap"][None, :]), Q / P.kj[None, :])


def classes(P, S, Vf, gap, T_, n):
    c = np.zeros((T_, n), np.int8); c[Vf < 0.6 * P.vf[None, :]] = 1; b = np.zeros((T_, n), bool); b[S["rb"], S["cb"]] = True; c[b] = 2
    for t, _ in S["wl"]:
        c[t + 1:t + 4][gap[t + 1:t + 4]] = 3; c[t + 4:t + 16][gap[t + 4:t + 16]] = 4; c[t + 16:t + 19][gap[t + 16:t + 19]] = 5
    return c


for p in PANELS:
    S = ST[p]; P, Pg = harness(p, S); val = P.ramp_validity(); V0, Q0 = S["Vb"], S["Qp"]; Vf, Qf = post(P, S, V0, Q0); gap = S["gap"]; day = np.arange(P.T) // 288; tv = Pg.target & np.isfinite(P.V) & np.isfinite(P.Q) & P.E & np.isfinite(V0)
    u = np.load(UH + f"{p}_unet{SET}.npz"); VU = np.clip(u["VU"].astype(float), 3.0, 1.15 * P.vf[None, :]); QU = np.clip(u["QU"].astype(float), 60.0, None); c = classes(P, S, Vf, gap, P.T, P.n); rg = np.unique(P.regime, return_inverse=True)[1].reshape(P.regime.shape)
    key = c.astype(int) * 10 + rg[:, None]; w0 = np.where(gap, 0.1, 0.6); Vfix = np.where(tv, (1 - w0) * Vf + w0 * VU, Vf); Qfix = np.where(tv, (1 - w0) * Qf + w0 * QU, Qf); Vfix = clip(P, S, Vfix, Qfix); R[p] = {}
    for par in (0, 1):
        fit = tv & ((day % 2 != par)[:, None]); Wv = w0.copy(); Wq = w0.copy(); tab = {}
        for k in np.unique(key[fit]):
            m = fit & (key == k); dv = (VU - Vf)[m]; dq = (QU - Qf)[m]; wv = float(np.clip((dv * (P.V - Vf)[m]).sum() / max((dv * dv).sum(), 1e-9), 0, 1)); wq = float(np.clip((dq * (P.Q - Qf)[m]).sum() / max((dq * dq).sum(), 1e-9), 0, 1)); Wv[key == k] = wv; Wq[key == k] = wq; tab[int(k)] = (round(wv, 2), round(wq, 2), int(m.sum()))
        Vm = np.where(tv, (1 - Wv) * Vf + Wv * VU, Vf); Qm = np.where(tv, (1 - Wq) * Qf + Wq * QU, Qf); Vm = clip(P, S, Vm, Qm); Pe = copy.copy(Pg); Pe.target = Pg.target & ((day % 2 == par)[:, None])
        a = (state_score(Pe, Vfix, Qfix), physics_score(Pe, Vfix, Qfix, validity=val)[2]); b = (state_score(Pe, Vm, Qm), physics_score(Pe, Vm, Qm, validity=val)[2]); R[p][par] = (b[0] - a[0], b[1] - a[1]); WT[(p, par)] = tab
    print(p, "read odd / even", [round(0.35 * R[p][q][0] + 0.15 * R[p][q][1], 5) for q in (0, 1)], flush=True); load.cache_clear()
pickle.dump(dict(R=R, WT=WT), open(WORK + "/t1/g178_table.pkl", "wb")); PN = list(PANELS)
for par, nm in ((1, "fitted on odd days, read on even days (the line)"), (0, "reverse")):
    v = np.array([0.35 * R[p][par][0] + 0.15 * R[p][par][1] for p in PN]); print(f"{nm}: {v.mean():+.5f} (SE {v.std(ddof=1) / np.sqrt(len(v)):.5f}) | dS_state {np.mean([R[p][par][0] for p in PN]):+.5f} dS_phys {np.mean([R[p][par][1] for p in PN]):+.5f} | panels > 0: {(v > 0).sum()}/10")
print("mean weights by class (speed, flow):", {c_: (round(float(np.mean([t[c_ * 10 + r][0] for t in WT.values() for r in range(3) if c_ * 10 + r in t])), 2), round(float(np.mean([t[c_ * 10 + r][1] for t in WT.values() for r in range(3) if c_ * 10 + r in t])), 2)) for c_ in range(6)})
G = pickle.load(open(WORK + "/t1/g177_eval.pkl", "rb")); obj = lambda par, key: np.array([0.35 * G[p][(par, *key)][0] + 0.15 * G[p][(par, *key)][1] for p in PN]); KEYS = sorted({k[1:] for k in G[PN[0]]}, key=str)
print("#127 arms on the same set, mean of the halves against the chain alone:"); [print(f"  {k}: odd {obj(0, k).mean():+.5f} even {obj(1, k).mean():+.5f}") for k in KEYS if k[0] == SET and (k[1] == "before" or k[1].startswith("fd")) and (k[1] != "before" or k[2] in (0.3, 0.6))]
print("  reference after 0.3 0:", round(obj(0, (SET, "after", 0.3, 0.0)).mean(), 5), round(obj(1, (SET, "after", 0.3, 0.0)).mean(), 5))
