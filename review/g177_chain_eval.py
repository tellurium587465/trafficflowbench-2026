"""G177 (11/01, ledgers #118b, #127): the image network mixed on the Feb harness through the production chain of B29 (ceiling, density clip, FD 0.75, boundary model), not the bare base of G168. Base = f3big + gapml_cv -> chain, as audit/b29_stage2_eval.py; the
first stage is read from its cache (work/t1/b29). Net = C:/tfb_data/tfb_nn/unet_h/<panel>_unet<set>.npz (g168 infer). Arms:
  after  : the production order (g169): chain, then (1 - w) chain + w net on the target cells outside / inside the blanks, ceiling and density clip
  before : (1 - w) first stage + w net, then the chain
  fd<a>  : `after`, then one more FD step on predicted congested cells at strength a (a_in inside the blanks)
Reads 0.35 dS_state + 0.15 dS_phys per panel on odd and on even days of February, against the chain alone. Choice on odd days, read on even days, and the reverse.
usage: py -B -X utf8 g177_chain_eval.py <sets, e.g. "",_s23,_all>"""
import os, sys, pickle, copy
SETS = (sys.argv[1:] or [""])[0].split(",")
src = open(r"C:/Users/user/tfb/audit/b29_stage2_eval.py", encoding="utf-8").read().split('D = pd.read_parquet(WORK + "/t1/b28_rows.parquet")')[0]
sys.argv = sys.argv[:1]; sys.dont_write_bytecode = True; exec(compile(src, "b29", "exec"), globals())
UH = "C:/tfb_data/tfb_nn/unet_h/"; ST = {p: stage1(p) for p in PANELS}; print("first stage ready", flush=True)
WO = (0.3, 0.4, 0.5, 0.6, 0.7); WI = (0.0, 0.1, 0.2, 0.3); FA = (0.5, 0.75, 1.0)
if os.environ.get("G177_IN"):
    WO = (0.7,); WI = (0.1, 0.3, 0.5, 0.7)


def clip(P, S, V, Q):
    return np.maximum(np.minimum(V, S["cap"][None, :]), Q / P.kj[None, :])


def fd(P, V, Q, tv, a):
    cong = V < 0.6 * P.vf[None, :]; k = np.maximum(P.kj[None, :] - Q / P.w[None, :], 1e-3); return np.where(cong & tv, (1 - a) * V + a * Q / k, V)


R = {}
for p in PANELS:
    S = ST[p]; P, Pg = harness(p, S); val = P.ramp_validity(); V0, Q0 = S["Vb"], S["Qp"]; Vf, Qf = post(P, S, V0, Q0); gap = S["gap"]; day = np.arange(P.T) // 288; tv = Pg.target & np.isfinite(P.V) & np.isfinite(P.Q) & P.E & np.isfinite(V0); R[p] = {}
    PE = []
    for par in (0, 1):
        Pe = copy.copy(Pg); Pe.target = Pg.target & ((day % 2 == par)[:, None]); PE.append((Pe, state_score(Pe, Vf, Qf), physics_score(Pe, Vf, Qf, validity=val)[2]))

    def rec(key, Vm, Qm):
        for par, (Pe, a0, b0) in enumerate(PE):
            R[p][(par, *key)] = (state_score(Pe, Vm, Qm) - a0, physics_score(Pe, Vm, Qm, validity=val)[2] - b0)

    n = 0
    for st in SETS:
        u = np.load(UH + f"{p}_unet{st}.npz"); VU = np.clip(u["VU"].astype(float), 3.0, 1.15 * P.vf[None, :]); QU = np.clip(u["QU"].astype(float), 60.0, None)
        if os.environ.get("G177_IN"):                                  # ledger #153: inside the blanks the net of this set
            u2 = np.load(UH + f"{p}_unet{os.environ['G177_IN']}.npz"); VU = np.where(gap, np.clip(u2["VU"].astype(float), 3.0, 1.15 * P.vf[None, :]), VU); QU = np.where(gap, np.clip(u2["QU"].astype(float), 60.0, None), QU)
        for wo in WO:
            for wi in WI:
                w = np.where(gap, wi, wo); Vm = np.where(tv, (1 - w) * Vf + w * VU, Vf); Qm = np.where(tv, (1 - w) * Qf + w * QU, Qf); rec((st, "after", wo, wi), clip(P, S, Vm, Qm), Qm); n += 1
                if wi == 0.0:
                    V1 = np.where(tv, (1 - w) * V0 + w * VU, V0); Q1 = np.where(tv, (1 - w) * Q0 + w * QU, Q0); rec((st, "before", wo, wi), *post(P, S, V1, Q1)); n += 1
                    if wo == 0.3:
                        for fa in FA:
                            for fi in FA:
                                aa = np.where(gap, fi, fa); rec((st, f"fd{fa}_{fi}", wo, wi), clip(P, S, fd(P, clip(P, S, Vm, Qm), Qm, tv, aa), Qm), Qm); n += 1
    print(p, n, "arms", flush=True); load.cache_clear()
pickle.dump(R, open(WORK + "/t1/g177_eval.pkl", "wb"))
PN = list(PANELS); obj = lambda par, key: np.array([0.35 * R[p][(par, *key)][0] + 0.15 * R[p][(par, *key)][1] for p in PN]); KEYS = sorted({k[1:] for k in R[PN[0]]}, key=str)
ref = (SETS[0], "after", 0.3, 0.0); print("reference (first set, after, 0.3, 0): odd", round(obj(0, ref).mean(), 5), "even", round(obj(1, ref).mean(), 5))
for sel, rd, nm in ((0, 1, "chosen on odd days, read on even days (the line)"), (1, 0, "reverse (record)")):
    b = max(KEYS, key=lambda c: obj(sel, c).mean()); v = obj(rd, b); r_ = obj(rd, ref); dt = np.mean([R[p][(rd, *b)][0] for p in PN]); dp = np.mean([R[p][(rd, *b)][1] for p in PN])
    print(f"{nm}: {b} | selection half {obj(sel, b).mean():+.5f} | read half {v.mean():+.5f} (SE {v.std(ddof=1) / np.sqrt(len(v)):.5f}) | reference on the read half {r_.mean():+.5f} | dS_state {dt:+.5f} dS_phys {dp:+.5f} | panels > 0: {(v > 0).sum()}/10 | better than the reference: {(v > r_).sum()}/10", flush=True)
print("top arms by the mean of both halves:")
for k in sorted(KEYS, key=lambda c: -(obj(0, c).mean() + obj(1, c).mean()))[:14]:
    print(f"  {k}: odd {obj(0, k).mean():+.5f} even {obj(1, k).mean():+.5f} panels>0 {((obj(0, k) + obj(1, k)) > 0).sum()}/10")
