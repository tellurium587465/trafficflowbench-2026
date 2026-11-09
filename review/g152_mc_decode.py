"""G152 (11/01, ledger #114, cross audit): onset set size by the exact expected IoU with the scoring mask. The candidates stay the top k of the
T+30 probabilities. For each k, Y ~ Bern(p) and E ~ Bern(e) are drawn together (2,000 draws, conditioned on Y and E meeting, as an onset window
always has an eligible queue at T+30) and the mean of |S & Y & E| / |(S | Y) & E| is computed; the best k is taken. best_set instead takes the
ratio E[I] / E[U] and ignores the mask (t2_model.py). e is fixed per instrument: M3 dev 0.585 (observed share of the masked layer), training
months the panel's eligible share over the other months. Only the T+30 row is scored on onset, so earlier rows are left empty.
  M3 dev  : the final onset flow (q24f 8 seeds, physics onset with adaptive a, evidence x0.25), k expectation, against best_set (A), identity to g108
  Jan, Feb: raw OOF (q25_nomem seeds 3, 4), raw best_set against the same decoder
usage: TFB_M3VER=v3 py -B -X utf8 g152_mc_decode.py [e_shift]"""
import os, sys, pickle
R = r"C:/Users/user/tfb/review/"; sys.dont_write_bytecode = True; SHIFT = float((sys.argv[1:] or ["0"])[0])
src = open(R + "g119_month_postproc.py", encoding="utf-8").read().split('if __name__ == "__main__":')[0]
sys.argv = [sys.argv[0], "run"]; exec(compile(src, "g119", "exec"), globals())
import importlib.util
sp = importlib.util.spec_from_file_location("g120", R + "g120_onset_gate.py"); g2 = importlib.util.module_from_spec(sp); sp.loader.exec_module(g2)
import t2_m3v2_gate as V
BEST = V.best_set; N = 2000; rng = [np.random.default_rng(114)]; E_NOW = [0.585]


def mc_set(p, elig):
    """p = the (6, n) probabilities raveled. Returns the raveled selection: the top k of row 5 with k from the exact expected IoU."""
    n = p.size // 6; p5 = p[5 * n:]; order = np.argsort(-p5)[:min(n, 80)]; r = rng[0]
    Y = r.random((N, n)) < p5; E = r.random((N, n)) < E_NOW[0]; YE = Y & E; keep = YE.any(axis=1)
    if keep.sum() < 50:
        return BEST(p, elig)
    YE, E = YE[keep], E[keep]; ny = YE.sum(axis=1); I = np.cumsum(YE[:, order], axis=1); Sz = np.cumsum(E[:, order], axis=1)
    U = Sz + ny[:, None] - I; v = (I / np.maximum(U, 1)).mean(axis=0); k = int(np.argmax(v)) + 1
    sel = np.zeros(p.size, bool); sel[5 * n + order[:k]] = True
    return sel


PN = g2.nn_blend(); phys = lgb.Booster(model_file=W + "physon_base.txt"); F_ = PO.rarity_table(); ev = PO.month_events("validation", F_); rows = []; size = []
E_NOW[0] = 0.585 + SHIFT
for w in g2.DEV:
    m = G.META[w]; i = G.POS[w]; Xw = np.asarray(G.X[G.OFF[i]:G.OFF[i + 1]]); ph = phys.predict(PO.feats(Xw)); a_ph = PO.adaptive_a(ev, m["day"]); A = MC = 0.0
    for k, pr in m3v2.kdist(m["day"]).items():
        p = PN[w].copy(); p[5] = (1 - a_ph) * p[5] + a_ph * ph
        V.best_set = BEST; s0 = V.onset_pred(p, m, "cur", k); A += pr * G.iou(s0, m)
        V.best_set = mc_set; s1 = V.onset_pred(p, m, "cur", k); MC += pr * G.iou(s1, m); size.append(("2031-03", int(s0[5].sum()), int(s1[5].sum())))
    rows.append(dict(w=w, cond=m["cond"], panel=m["panel"], day=m["day"], hour=(m["t"] % 288) // 12, A=A, MC=MC))
V.best_set = BEST; M = pd.DataFrame(rows); M.to_csv(W + "g152_mc.csv", index=False)
o = pd.read_csv(W + "g108_rectangles.csv"); o = o[o.w.isin(M.w)].set_index("w").A.reindex(M.w).to_numpy()
print(f"M3 dev onset {len(M)} windows | identity against g108 A: max |d| {np.abs(o - M.A.to_numpy()).max():.2e} | A {M.A.mean():.4f} MC {M.MC.mean():.4f} diff {M.MC.mean() - M.A.mean():+.4f} | windows changed {(M.A != M.MC).sum()}", flush=True)
OOF = [pickle.load(open(W + f"oof_q25_nomem_s{s}.pkl", "rb")) for s in (3, 4)]; sel = np.flatnonzero(T.META["is_sel"] & np.isin(MON, MONTHS) & (T.META["cond"] == "queue_onset"))
ALLON = np.flatnonzero(T.META["is_sel"] & (T.META["cond"] == "queue_onset")); PANELS_ = T.META["panel"]
rate = {}
for mo in MONTHS:
    for p in sorted({str(PANELS_[k]) for k in sel}):
        ks = [k for k in ALLON if str(PANELS_[k]) == p and MON[k] != mo]; rate[(mo, p)] = float(np.mean([T.LAB[k][1][5].mean() for k in ks]))
print("eligible share at T+30, other months: min", round(min(rate.values()), 3), "max", round(max(rate.values()), 3), flush=True)
tr = []
for k in sel:
    mo = str(MON[k]); p = str(PANELS_[k]); lab, el = T.LAB[k]; lab, el = lab.astype(bool), el.astype(bool); pb = (OOF[0][k] + OOF[1][k]) / 2; E_NOW[0] = rate[(mo, p)] + SHIFT
    s0 = BEST(pb.ravel(), np.ones(pb.size, bool)).reshape(pb.shape); s1 = mc_set(pb.ravel(), None).reshape(pb.shape)
    tr.append(dict(month=mo, panel=p, day=int(str(T.META["date"][k])[8:10]), raw=iou(s0, lab, el, "queue_onset"), MC=iou(s1, lab, el, "queue_onset"))); size.append((mo, int(s0[5].sum()), int(s1[5].sum())))
D = pd.DataFrame(tr); D.to_csv(W + "g152_mc_train.csv", index=False)
for mo, g in D.groupby("month"):
    d = (g.MC - g.raw).to_numpy(); ids = pd.factorize(g.panel + "_" + g.day.astype(str))[0]; B = np.bincount(ids, d); Nn = np.bincount(ids); rr = np.random.default_rng(0); Rm = rr.multinomial(len(B), np.full(len(B), 1 / len(B)), size=2000); s = ((Rm @ B) / (Rm @ Nn)).std()
    print(f"{mo}: {len(g)} windows raw {g.raw.mean():.4f} MC {g.MC.mean():.4f} diff {d.mean():+.4f} (SE {s:.4f}, z {d.mean() / s:+.2f}) changed {(d != 0).sum()}", flush=True)
Z = pd.DataFrame(size, columns=["month", "best_set", "mc"]); print(Z.groupby("month").mean().round(2).to_string())
