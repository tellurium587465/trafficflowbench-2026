"""G160 (11/01, ledger #117b): M3 dev with the speed inputs on the training scale (G159) against the same inputs unchanged, both by the CPU path. Onset: the 40 q24f networks, physics onset on
the same inputs, evidence x0.25, best_set (flow A) and the networks alone (flow B). Ongoing: 0.8 (0.8 net + 0.2 tfm 5 seeds) + 0.2 LightGBM -> cone -> seen x0.85 -> best_set -> far gate,
net = c24f_attnglob s0, s1 with no fine-tuning (X1 was fitted on unchanged inputs, so it stays out of this stage on both sides). Hold-out days are not read.
usage: TFB_M3VER=v3 py -B -X utf8 g160_vx_score.py"""
import os, sys, pickle, importlib.util
os.environ.setdefault("TFB_M3VER", "v3"); sys.dont_write_bytecode = True; R0 = r"C:/Users/user/tfb/"
import numpy as np, pandas as pd, lightgbm as lgb


def module(nm, path):
    spec = importlib.util.spec_from_file_location(nm, path); mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


sys.argv = [sys.argv[0], "run"]; g = module("g121", R0 + "review/g121_ongoing_oracle.py"); g132 = module("g132", R0 + "review/g132_p3_grid.py")
G, V, m3v2, best_set = g.G, g.V, g.m3v2, g.best_set
import t2_phys_onset as PO
from common import NN
ld = lambda f: pickle.load(open(g.W + f, "rb")); CF5 = (("attnglob", 2.0), ("attnwp05", 2.0), ("attn1", 2.0), ("drop1", 1.0), ("r13", 1.0)); ON = [(f"q24f_{c}_s{s}", wt) for s in range(8) for c, wt in CF5]; WS = sum(wt for _, wt in ON)
DIR = {a: f"{NN}/nn_test_m3v3{a}" for a in ("dv", "vx")}; XS = {a: np.load(d + "/X.npy", mmap_mode="r") for a, d in DIR.items()}; OF = np.load(DIR["dv"] + "/off.npy"); WD = [str(w) for w in np.load(DIR["dv"] + "/window_id.npy", allow_pickle=True)]; PS = {w: i for i, w in enumerate(WD)}
PRN = {a: {t: ld(f"probs_m3v3{a}__{t}.pkl") for t, _ in ON} for a in ("dv", "vx")}; PBN = {a: [ld(f"probs_m3v3{a}__c24f_attnglob_s{s}.pkl") for s in (0, 1)] for a in ("dv", "vx")}; PTN = {a: [ld(f"probs_m3v3{a}__tfm4_s{s}.pkl") for s in range(5)] for a in ("dv", "vx")}
phys = lgb.Booster(model_file=g.W + "physon_base.txt"); ev = PO.month_events("validation", PO.rarity_table()); cols = {c: {} for c in ("A0", "A1", "B0", "B1", "O0", "O1")}
Xw_ = lambda a, w: np.array(XS[a][OF[PS[w]]:OF[PS[w] + 1]])
for w in WD:
    m = G.META[w]; kd = m3v2.kdist(m["day"])
    if m["cond"] == "queue_onset":
        a_ph = PO.adaptive_a(ev, m["day"])
        for a, nA, nB in (("dv", "A0", "B0"), ("vx", "A1", "B1")):
            pn = sum(wt * PRN[a][t][w] for t, wt in ON) / WS; ph = phys.predict(PO.feats(Xw_(a, w))); p = pn.copy(); p[5] = (1 - a_ph) * p[5] + a_ph * ph
            cols[nA][w] = sum(pr * G.iou(V.onset_pred(p, m, "cur", k), m) for k, pr in kd.items()); cols[nB][w] = sum(pr * G.iou(V.onset_pred(pn, m, "cur", k), m) for k, pr in kd.items())
    else:
        i = G.POS[w]; lq = np.asarray(G.X[G.OFF[i]:G.OFF[i + 1], G.CI], float) == 1; P, _, q = m3v2._panel(m["panel"]); t = m["t"]; d0 = t - (t % 288) + 60; _, cone, _ = g.g55.parts(w); nr = g132.nears(lq, P.n)[3]
        pre = {a: (0.8 * (0.8 * (PBN[a][0][w] + PBN[a][1][w]) / 2 + 0.2 * sum(d[w] for d in PTN[a]) / 5) + 0.2 * g.FS.lgb_ongoing(Xw_(a, w))) for a in ("dv", "vx")}; sc = dict(dv=0.0, vx=0.0)
        for k, pr in kd.items():
            hid = V._hidden_today(m, k); rr = [r for r in range(d0, t) if r not in hid]; ev_ = q[rr].any(axis=0) if rr else np.zeros(P.n, bool); s2, _ = m3v2.seen(m["panel"], t, k, -18, 12)
            ex = G.dil(ev_ | s2, 2); s, nd = m3v2.seen(m["panel"], t, k, -18, 12); sw = np.where(G.dil(s, 1), 1.0, 0.85)[None, :] if nd else 1.0; cf = np.where(ex[None, :], 1.0, np.where(cone < 0.999, 0.5, 1.0))
            for a in sc:
                pb = pre[a] * cf * sw; sc[a] += pr * G.iou(g132.gate(best_set(pb.ravel(), np.ones(pb.size, bool)).reshape(pb.shape), ex, nr), m)
        cols["O0"][w] = sc["dv"]; cols["O1"][w] = sc["vx"]
D = V.frame(cols).dropna(subset=["A0", "O0"], how="all").reset_index(drop=True); D.to_csv(g.W + "g160_vx.csv", index=False)
on = D[D.cond == "queue_onset"]; og = D[D.cond == "queue_ongoing"]; ref = pd.read_csv(g.W + "g108_rectangles.csv").set_index("w").A
print(f"onset {len(on)}: A CPU path {on.A0.mean():.4f} (g108 {ref.loc[on.w].mean():.4f}) -> corrected {on.A1.mean():.4f} | B {on.B0.mean():.4f} -> {on.B1.mean():.4f} | ongoing {len(og)}: {og.O0.mean():.4f} -> {og.O1.mean():.4f}", flush=True)
