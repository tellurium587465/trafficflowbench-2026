"""G172 (11/01, ledger #123, stage 0, strategy T2_DEEP_REVIEW 8-2 B1): does the row at T, which the finals drop, add to the ongoing probabilities? The rules allow it (742068). State of link j and of j - 1, j + 1 in row T of the released layer:
queued, free or hidden. Cells: all eligible cells of every horizon. p: final A's mix on M3 dev (0.8 (0.8 X1 + 0.2 tfm) + 0.2 LightGBM, k expectation, before the cone), the OOF of q25_nomem seeds 3, 4 on January and February.
Logistic y ~ logit(p) + h against + queued and free flags of the three links x horizon group (T+5..10, T+15..20, T+25..30), five folds by window, log loss per cell, window bootstrap.
Line (declared first): improvement >= 2 SE with the same sign on M3 dev and on January + February. Otherwise closed. Hold-out days are not read.
usage: TFB_M3VER=v3 py -B -X utf8 g172_t_row.py"""
import os, sys, pickle
R = r"C:/Users/user/tfb/review/"; sys.dont_write_bytecode = True
src = open(R + "g119_month_postproc.py", encoding="utf-8").read().split('if __name__ == "__main__":')[0]
sys.argv = [sys.argv[0], "run"]; exec(compile(src, "g119", "exec"), globals())
import t2_final_spec as FS
from sklearn.linear_model import LogisticRegression
XA = [pickle.load(open(W + f"probs_m3{G.VER}v2x1__c24f_attnglob_s{s}.pkl", "rb")) for s in (0, 1)]; X1 = {key: (XA[0][key] + XA[1][key]) / 2 for key in XA[0]}; PL = [pickle.load(open(W + f"probs_m3v3__tfm4_s{s}.pkl", "rb")) for s in range(5)]
lg_ = lambda x: np.log(np.clip(x, 1e-5, 1 - 1e-5) / (1 - np.clip(x, 1e-5, 1 - 1e-5)))


def cells(w, inst, p, y, e, row, vcut, rows):
    n = len(row); f = np.isfinite(row); q = f & (row <= vcut); fr = f & ~q; sh = lambda a, o: np.r_[np.zeros(max(o, 0), bool), a[:n - abs(o)], np.zeros(max(-o, 0), bool)]; ii, jj = np.nonzero(e)
    if len(ii):
        rows.append(pd.DataFrame(dict(inst=inst, w=str(w), h=ii.astype(np.int8), lp=lg_(p[ii, jj]).astype(np.float32), y=y[ii, jj].astype(np.int8), q0=q[jj], f0=fr[jj], qu=sh(q, 1)[jj], fu=sh(fr, 1)[jj], qd=sh(q, -1)[jj], fd=sh(fr, -1)[jj])))


rows = []; DEV = [w for w in G.WID if G.META[w]["cond"] == "queue_ongoing" and not m3v2.is_holdout(np.array([G.META[w]["day"]]))[0]]
for w in DEV:
    m = G.META[w]; i = G.POS[w]; Xw = np.asarray(G.X[G.OFF[i]:G.OFF[i + 1]]); P, _, _ = m3v2._panel(m["panel"]); pl = FS.lgb_ongoing(Xw); kd = m3v2.kdist(m["day"]); pt = sum(d[w] for d in PL) / 5
    p = sum(pr * (0.8 * (0.8 * X1[(w, k)] + 0.2 * pt) + 0.2 * pl) for k, pr in kd.items()); cells(w, "2031-03", p, m["y"].T.astype(bool), m["m"].T.astype(bool), P.V[m["t"]], P.vcut, rows)
print("March dev windows", len(DEV), flush=True)
OOF = [pickle.load(open(W + f"oof_q25_nomem_s{s}.pkl", "rb")) for s in (3, 4)]
for mo in MONTHS:
    ks = [k for k in np.flatnonzero(T.META["is_sel"] & (MON == mo)) if str(T.META["cond"][k]) == "queue_ongoing"]
    for pn in sorted({str(T.META["panel"][k]) for k in ks}):
        P = load(pn, "train"); Vv = np.where(P.blank, np.nan, P.V)
        for k in [k for k in ks if str(T.META["panel"][k]) == pn]:
            t = int(T.META["t"][k]); lab, el = T.LAB[k]; cells(k, mo, (OOF[0][k] + OOF[1][k]) / 2, lab.astype(bool), el.astype(bool), Vv[t], P.vcut, rows)
        load.cache_clear()
    print(mo, len(ks), flush=True)
D = pd.concat(rows, ignore_index=True); D.to_parquet(W + "g172_t_cells.parquet", index=False)
print(D.groupby("inst").agg(cells=("y", "size"), windows=("w", "nunique"), y=("y", "mean"), q0=("q0", "mean"), f0=("f0", "mean")).round(3).to_string())
print("h = 5, actual against predicted by the link's state at T"); x = D[D.h == 5].assign(st=lambda d: np.where(d.q0, "queued", np.where(d.f0, "free", "hidden")), p=lambda d: 1 / (1 + np.exp(-d.lp))); print(x.groupby(["inst", "st"]).agg(cells=("y", "size"), y=("y", "mean"), p=("p", "mean")).round(4).to_string())


def design(g, ext):
    H = np.eye(6)[g.h.to_numpy()][:, 1:]; X = [g.lp.to_numpy()[:, None], H]
    if ext:
        G3 = np.eye(3)[np.minimum(g.h.to_numpy() // 2, 2)]; S = g[["q0", "f0", "qu", "fu", "qd", "fd"]].to_numpy(float); X += [S[:, :, None][:, :, 0] * G3[:, [a]] for a in range(3)]
    return np.hstack(X)


def test(g, name):
    ws = g.w.unique(); fold = dict(zip(ws, np.random.default_rng(0).integers(5, size=len(ws)))); f = g.w.map(fold).to_numpy(); y = g.y.to_numpy(); ll = np.zeros((len(g), 2))
    for e_, ext in enumerate((False, True)):
        X = design(g, ext)
        for k in range(5):
            mdl = LogisticRegression(C=1e4, max_iter=300).fit(X[f != k], y[f != k]); pr = np.clip(mdl.predict_proba(X[f == k])[:, 1], 1e-6, 1 - 1e-6); ll[f == k, e_] = -(y[f == k] * np.log(pr) + (1 - y[f == k]) * np.log(1 - pr))
    ids = pd.factorize(g.w)[0]; d = np.bincount(ids, ll[:, 0] - ll[:, 1]); nc = np.bincount(ids); Rm = np.random.default_rng(0).multinomial(len(d), np.full(len(d), 1 / len(d)), size=2000); s = ((Rm @ d) / (Rm @ nc)).std(); m_ = d.sum() / nc.sum()
    print(f"{name}: {len(ws)} windows, {len(g)} cells | log loss base {ll[:, 0].mean():.5f} ext {ll[:, 1].mean():.5f} | improvement {m_:+.6f} (SE {s:.6f}, z {m_ / s:+.2f})", flush=True)


test(D[D.inst == "2031-03"], "March dev"); test(D[D.inst.isin(MONTHS)], "January + February")
for mo in MONTHS:
    test(D[D.inst == mo], mo)
