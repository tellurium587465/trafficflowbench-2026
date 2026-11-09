"""G142 (10/31, ledger #101-#103): score G141's arms on M3 v3 DEV windows (held-out days are not read), the mean of c24f_attnglob s0 / s1, k expectation with the simulated blanks.
Ongoing columns: A = final A6's flow (0.8 (0.8 X1 + 0.2 tfm4 5 seeds) + 0.2 LightGBM -> cone -> seen x0.85 -> best_set -> far gate) with the saved X1; one column per arm with its X1 instead; a5_* = the same without tfm; raw_* = best_set on the X1 alone.
Onset columns: A = the flow of G110's column A (q24f CF5 blend, physics onset, evidence rule); tta3 / tta5 = the same with the blend's networks averaged over the flow scales.
Identity first: A against g110_tfm5.csv (ongoing T, onset A). Writes work/t2/x1_arms/g142.csv for judge.py --cond --kind model; an arm without both seeds' parts is left out.
usage: TFB_M3VER=v3 py g142_x1_arms_score.py"""
import os, sys, glob, pickle
os.environ.setdefault("TFB_M3VER", "v3")
sys.argv = [sys.argv[0]]
src = open(r"C:/Users/user/tfb/review/g108_rectangles.py", encoding="utf-8").read().split("\ncols = {c: {} for c")[0]
exec(compile(src, "g108", "exec"))
OUT = W + "x1_arms/"; TAGS = [f"c24f_attnglob_s{s}" for s in (0, 1)]; G3 = (0.9, 1.0, 1.1); G5 = (0.8, 0.9, 1.0, 1.1, 1.2)


def arm(name):
    PR = []
    for t in TAGS:
        fs = sorted(glob.glob(OUT + f"parts/{name}__{t}__*.pkl")); done = os.path.exists(OUT + f"{name}__{t}.log") and "ALLDONE" in open(OUT + f"{name}__{t}.log").read()
        if not (fs and done):
            return None
        d = {}
        for f in fs:
            d.update(pickle.load(open(f, "rb")))
        PR.append(d)
    return {key: (PR[0][key] + PR[1][key]) / 2 for key in PR[0]}


_PT = [ld(f"probs_{PRE}__tfm4_s{s}.pkl") for s in range(5)]; PT = {w: sum(d[w] for d in _PT) / 5 for w in _PT[0]}
raw = lambda p: best_set(p.ravel(), np.ones(p.size, bool)).reshape(p.shape)
AR = {n: arm(n) for n in ("sd", "hl3", "hl7", "pl")}; AR = {n: a for n, a in AR.items() if a}; TT = arm("tta"); print("arms", list(AR), "tta", TT is not None, flush=True)
if TT:
    AR["tta3"] = {(w, k): sum(TT[(w, k, s)] for s in G3) / 3 for (w, k, s) in TT if s == 1.0}; AR["tta5"] = {(w, k): sum(TT[(w, k, s)] for s in G5) / 5 for (w, k, s) in TT if s == 1.0}
    print("identity tta scale 1 against the saved X1: max |dp|", max(float(np.abs(TT[(w, k, 1.0)] - X1[(w, k)]).max()) for (w, k, s) in TT if s == 1.0), flush=True)
TN = {t: pickle.load(open(OUT + f"parts/tton__{t}.pkl", "rb")) for t, _ in ON if os.path.exists(OUT + f"parts/tton__{t}.pkl")}; TN = TN if len(TN) == len(ON) else {}; print("onset networks at the scales", len(TN), flush=True)
cols = {c: {} for c in ["A", "a5_A", "raw_A"] + [p + a for a in AR for p in ("", "a5_", "raw_")] + (["tta3", "tta5"] if TN and not AR.get("tta3") else [])}
wsum = sum(wt for _, wt in ON); dev = [(i, w) for i, w in enumerate(G.WID) if not m3v2.is_holdout(np.array([G.META[w]["day"]]))[0]]; dm = 0.0
for i, w in dev:
    m = G.META[w]; kd = m3v2.kdist(m["day"]); Xw = np.asarray(G.X[G.OFF[i]:G.OFF[i + 1]])
    if m["cond"] == "queue_ongoing":
        pl = FS.lgb_ongoing(Xw); cf = {k: cone_f(m, k) for k in kd}
        def score(P):
            return (sum(pr * G.iou(og_pred((0.8 * (0.8 * P[(w, k)] + 0.2 * PT[w]) + 0.2 * pl) * cf[k], m, k, None, None), m) for k, pr in kd.items()), sum(pr * G.iou(og_pred((0.8 * P[(w, k)] + 0.2 * pl) * cf[k], m, k, None, None), m) for k, pr in kd.items()),
                    sum(pr * G.iou(raw(P[(w, k)]), m) for k, pr in kd.items()))
        for nm, P in [("A", X1)] + list(AR.items()):
            a, b, c = score(P); cols[nm][w] = a; cols["a5_" + nm][w] = b; cols["raw_" + nm][w] = c
    elif TN:
        ph = phys.predict(PO.feats(Xw)); a_ph = PO.adaptive_a(ev, m["day"])
        def onset(sc):
            pn = sum(wt * sum(TN[t][s][w] for s in sc) / len(sc) for t, wt in ON) / wsum; p = pn.copy(); p[5] = (1 - a_ph) * p[5] + a_ph * ph
            return sum(pr * G.iou(V.onset_pred(p, m, "cur", k), m) for k, pr in kd.items())
        cols["A"][w] = onset((1.0,))
        for nm, sc in (("tta3", G3), ("tta5", G5)):
            cols[nm][w] = onset(sc)
D = V.frame(cols); D.to_csv(OUT + "g142.csv", index=False); R0 = pd.read_csv(W + "g110_tfm5.csv").set_index("w")
d = D.dropna(subset=["A"]); og = d.cond == "queue_ongoing"; ref = np.where(og, R0.loc[d.w, "T"].to_numpy(), R0.loc[d.w, "A"].to_numpy())
print("windows", og.sum(), (~og).sum(), "identity A against g110_tfm5.csv: max |d| ongoing", float(np.abs(d.A.to_numpy() - ref)[og].max()), "onset", float(np.abs(d.A.to_numpy() - ref)[~og].max()) if (~og).any() else None, flush=True)
