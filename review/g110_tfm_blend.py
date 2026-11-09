"""G110 (10/29, strategy lane 9, ledger #55): the attention-backbone network (tfm4, trained on Kaggle GPU, FINAL on the
training months) blended into the final A flow on M3 v3 (G93 / G108 harness, k expectation with the simulated blanks).
  onset   : p' = 0.8 p_CF5(q24f 8 seeds) + 0.2 p_tfm, then the physics onset mix (month-adaptive a), evidence rule, decoder
  ongoing : p = 0.8 (0.8 X1 + 0.2 p_tfm) + 0.2 LightGBM -> cone -> evidence weight / best_set / far gate (no X1 for tfm yet)
Platform check (before reading the blend): the Kaggle retrain of the production q24f_attnglob_s0 line (kq24f_attnglob_s0,
T4 fp16) against the local bf16 model, each used alone (onset: alone in the onset flow; ongoing: its raw probabilities
-> cone -> decoder), with the local seeds 1-3 of the same configuration as the seed spread.
Also: correlation of the residuals (p - y on the scored cells) between tfm and the q24f blend.
Probabilities: work/t2/probs_m3v3__<tag>.pkl from t2_nn_cpu_infer.py <tag> nn_test_m3v3 m3v3.
Control (10/29 10:31, before the tfm result): C = the same 0.2 mix with the Kaggle q24f retrain (a new seed of the family);
the primary comparison is T vs C (diversity), T vs A alongside.
Writes work/t2/g110_tfm_blend.csv (columns A, T, C) for judge.py --cond.
usage: TFB_M3VER=v3 py g110_tfm_blend.py [tfm tag=tfm4_s0] [kaggle q24f tag=kq24f_attnglob_s0]"""
import os, sys
os.environ.setdefault("TFB_M3VER", "v3")
TFM = sys.argv[1] if len(sys.argv) > 1 else "tfm4_s0"; KQ = sys.argv[2] if len(sys.argv) > 2 else "kq24f_attnglob_s0"
sys.argv = [sys.argv[0]]
src = open(r"C:/Users/user/tfb/review/g108_rectangles.py", encoding="utf-8").read().split("\ncols = {c: {} for c")[0]
exec(compile(src, "g108", "exec"))                              # G93's final A flows, og_pred(), cone_f(), PRO, X1, phys
_PT = [ld(f"probs_{PRE}__{t}.pkl") for t in TFM.split(",")]                 # 10/29 #56: a comma list = the seed mean
PT = {w: sum(d[w] for d in _PT) / len(_PT) for w in _PT[0]}; PK = ld(f"probs_{PRE}__{KQ}.pkl")
LOC = {s: ld(f"probs_{PRE}__q24f_attnglob_s{s}.pkl") for s in range(4)}

cols = {c: {} for c in ("A", "T", "C")}; plat = {c: {} for c in ["kaggle"] + [f"local_s{s}" for s in range(4)]}; dmax = 0.0
res_t, res_q = [], []
for i, w in enumerate(G.WID):
    m = G.META[w]; Xw = np.asarray(G.X[G.OFF[i]:G.OFF[i + 1]]); og = m["cond"] == "queue_ongoing"
    kd = m3v2.kdist(m["day"]); y, mk = m["y"].T.astype(bool), m["m"].T.astype(bool)
    if og:
        pl = FS.lgb_ongoing(Xw); sa = st = sc_ = 0.0; spl = {c: 0.0 for c in plat}
        for k, pr in kd.items():
            cf = cone_f(m, k)
            sa += pr * G.iou(og_pred((0.8 * X1[(w, k)] + 0.2 * pl) * cf, m, k, None, None), m)
            st += pr * G.iou(og_pred((0.8 * (0.8 * X1[(w, k)] + 0.2 * PT[w]) + 0.2 * pl) * cf, m, k, None, None), m)
            sc_ += pr * G.iou(og_pred((0.8 * (0.8 * X1[(w, k)] + 0.2 * PK[w]) + 0.2 * pl) * cf, m, k, None, None), m)
            for c, P_ in [("kaggle", PK)] + [(f"local_s{s}", LOC[s]) for s in range(4)]:
                spl[c] += pr * G.iou(og_pred(P_[w] * cf, m, k, None, None), m)
        res_t.append((PT[w] - y)[mk]); res_q.append((sum(LOC[s][w] for s in range(4)) / 4 - y)[mk])
    else:
        pn = sum(wt * PRO[t][w] for t, wt in ON) / sum(wt for _, wt in ON); ph = phys.predict(PO.feats(Xw)); a_ph = PO.adaptive_a(ev, m["day"])
        def onset(pbase):
            p = pbase.copy(); p[5] = (1 - a_ph) * p[5] + a_ph * ph
            return sum(pr * G.iou(V.onset_pred(p, m, "cur", k), m) for k, pr in kd.items())
        sa, st, sc_ = onset(pn), onset(0.8 * pn + 0.2 * PT[w]), onset(0.8 * pn + 0.2 * PK[w])
        spl = {"kaggle": onset(PK[w])} | {f"local_s{s}": onset(LOC[s][w]) for s in range(4)}
        res_t.append((PT[w][5] - y[5])[mk[5]]); res_q.append((pn[5] - y[5])[mk[5]])
    cols["A"][w] = sa; cols["T"][w] = st; cols["C"][w] = sc_
    for c in plat:
        plat[c][w] = spl[c]
    dmax = max(dmax, abs(sa - REF.loc[w, "lgb_cone" if og else "phys"]))
print(f"identity: A vs G93 (lgb_cone / phys): max |d| {dmax:.2e}")
OUTN = os.environ.get("G110_OUT", "g110_tfm_blend")                                 # 10/29 #56: g110_tfm5
D = V.frame(cols); D.to_csv(W + f"{OUTN}.csv", index=False); dv = ~m3v2.is_holdout(D.day)
Pl = V.frame(plat); Pl.to_csv(W + f"{OUTN}_platform.csv", index=False)
for cond in ("queue_onset", "queue_ongoing"):
    d = D[dv & (D.cond == cond)]; pl_ = Pl[dv & (Pl.cond == cond)]
    print(f"{cond} dev {len(d)}: A {d.A.mean():.4f} | with tfm 0.2 (T) {d['T'].mean():.4f} | with the Kaggle q24f 0.2 (C, control) {d.C.mean():.4f} | T - C {(d['T'] - d.C).mean():+.4f} | T - A {(d['T'] - d.A).mean():+.4f}")
    print("   platform (each model alone): " + " | ".join(f"{c} {pl_[c].mean():.4f}" for c in plat))
rt, rq = np.concatenate(res_t), np.concatenate(res_q)
print(f"residual correlation (tfm vs q24f seeds 0-3) on the scored cells: {np.corrcoef(rt, rq)[0, 1]:.3f} (cells {len(rt)})")
