"""G153 (11/01, ledger #113, stage 0, literature audit T2 ongoing): does the age of a jam beyond the visible hour add to the final probability?
Age: for each run of links queued at the last visible row, walk back from T-1 over the same day's rows at the run's head +- 1, stopping after two
rows in a row with every observed speed above 0.6 v_f. Hidden or unobserved rows are unknown: the walk goes on and the window is flagged. Cells: the
links of those runs, every horizon, labels on eligible cells. p: final A's mix on M3 dev (0.8 X1 + 0.2 LightGBM, k expectation, before the cone), the
OOF of q25_nomem seeds 3, 4 on January and February. Logistic y ~ logit(p) + h against + age bins (< 30, 30-60, 60-120, >= 120 min) + the flag,
five folds by window. The improvement in log loss per cell, window bootstrap.
Line (declared first): improvement >= 2 SE with the same sign on M3 dev and on January + February. Otherwise closed.
usage: TFB_M3VER=v3 py -B -X utf8 g153_jam_age.py"""
import os, sys, pickle
R = r"C:/Users/user/tfb/review/"; sys.dont_write_bytecode = True
src = open(R + "g119_month_postproc.py", encoding="utf-8").read().split('if __name__ == "__main__":')[0]
sys.argv = [sys.argv[0], "run"]; exec(compile(src, "g119", "exec"), globals())
import t2_m3v2_gate as V, t2_final_spec as FS
from sklearn.linear_model import LogisticRegression
VER = G.VER; XA = [pickle.load(open(W + f"probs_m3{VER}v2x1__c24f_attnglob_s{s}.pkl", "rb")) for s in (0, 1)]; X1 = {key: (XA[0][key] + XA[1][key]) / 2 for key in XA[0]}
lg_ = lambda x: np.log(np.clip(x, 1e-5, 1 - 1e-5) / (1 - np.clip(x, 1e-5, 1 - 1e-5)))


def runs(v):
    dd = np.diff(np.r_[0, np.asarray(v, int), 0]); return list(zip(np.flatnonzero(dd == 1), np.flatnonzero(dd == -1) - 1))


def age(V_, vf, t, a, b, hid, d0):
    n = V_.shape[1]; cols = list(range(max(0, b - 1), min(n, b + 2))); clear = 0; unk = False; r = t - 1
    while r >= d0:
        if r in hid or not np.isfinite(V_[r, cols]).any():
            unk = True; clear = 0 if os.environ.get("G153_RESET") == "1" else clear       # unknown rows are skipped; G153_RESET=1 = the first, stricter reading
        else:
            v = V_[r, cols]; o = np.isfinite(v)
            clear = clear + 1 if (v[o] > 0.6 * vf[cols][o]).all() else 0
            if clear == 2:
                return (t - 1 - (r + 1)) * 5, unk
        r -= 1
    return (t - 1 - d0) * 5, True


def cells(w, inst, p, y, e, lq, V_, vf, t, hid, rows):
    d0 = t - (t % 288)
    for a, b in runs(lq):
        ag, cens = age(V_, vf, t, a, b, hid, d0); ii, jj = np.nonzero(e[:, a:b + 1]); jj = jj + a
        if len(ii):
            rows.append(pd.DataFrame(dict(inst=inst, w=str(w), h=ii.astype(np.int8), lp=lg_(p[ii, jj]).astype(np.float32), y=y[ii, jj].astype(np.int8), age=float(ag), cens=int(cens))))


rows = []; DEV = [w for w in G.WID if G.META[w]["cond"] == "queue_ongoing" and not m3v2.is_holdout(np.array([G.META[w]["day"]]))[0]]
for w in DEV:
    m = G.META[w]; i = G.POS[w]; Xw = np.asarray(G.X[G.OFF[i]:G.OFF[i + 1]]); lq = np.asarray(Xw[:, G.CI], float) == 1; P, _, _ = m3v2._panel(m["panel"]); pl = FS.lgb_ongoing(Xw); kd = m3v2.kdist(m["day"])
    p = sum(pr * (0.8 * X1[(w, k)] + 0.2 * pl) for k, pr in kd.items()); hid = V._hidden_today(m, min(3, m["day"]))
    cells(w, "2031-03", p, m["y"].T.astype(bool), m["m"].T.astype(bool), lq, P.V, P.vf, m["t"], hid, rows)
print("March dev windows", len(DEV), flush=True)
OOF = [pickle.load(open(W + f"oof_q25_nomem_s{s}.pkl", "rb")) for s in (3, 4)]
for mo in MONTHS:
    ks = [k for k in np.flatnonzero(T.META["is_sel"] & (MON == mo)) if str(T.META["cond"][k]) == "queue_ongoing"]
    for pn in sorted({str(T.META["panel"][k]) for k in ks}):
        P = load(pn, "train"); key = (mo, pn, "mask"); CTX.clear(); sim_blank.cache_clear(); CTX[key] = Ctx(key, P, np.where(P.blank, np.nan, P.V), P.V); c = CTX[key]; Vv = np.where(P.blank, np.nan, P.V)
        for k in ks:
            if str(T.META["panel"][k]) != pn:
                continue
            t = int(T.META["t"][k]); dom = int(P.times[t][8:10]) - 1; lab, el = T.LAB[k]; Xw = np.asarray(XN[OFFN[k]:OFFN[k + 1]]); lq = np.asarray(Xw[:, G.CI], float) == 1; Vw = Vv.copy() if False else Vv
            save = Vw[t - 12:t].copy(); Vw[t - 12:t] = P.V[t - 12:t]; hid = sim_blank(key, t // 288, min(3, dom), t)
            cells(k, mo, (OOF[0][k] + OOF[1][k]) / 2, lab.astype(bool), el.astype(bool), lq, Vw, P.vf, t, hid, rows); Vw[t - 12:t] = save
        load.cache_clear()
    print(mo, len(ks), flush=True)
D = pd.concat(rows, ignore_index=True); D["bin"] = np.digitize(D.age, [30, 60, 120]); D.to_parquet(W + "g153_age_cells.parquet", index=False)
print(D.groupby("inst").agg(cells=("y", "size"), windows=("w", "nunique"), y=("y", "mean"), age=("age", "median"), cens=("cens", "mean")).round(3).to_string())
print(D[D.h == 5].groupby(["inst", "bin"]).agg(cells=("y", "size"), y=("y", "mean"), p=("lp", lambda s: float((1 / (1 + np.exp(-s))).mean()))).round(3).to_string())


def design(g, ext):
    H = np.eye(6)[g.h.to_numpy()][:, 1:]; X = [g.lp.to_numpy()[:, None], H]
    if ext:
        X += [np.eye(4)[g["bin"].to_numpy()][:, 1:], g.cens.to_numpy()[:, None]]
    return np.hstack(X)


def test(g, name):
    ws = g.w.unique(); fold = dict(zip(ws, np.random.default_rng(0).integers(5, size=len(ws)))); f = g.w.map(fold).to_numpy(); y = g.y.to_numpy(); ll = np.zeros((len(g), 2))
    for e_, ext in enumerate((False, True)):
        X = design(g, ext)
        for k in range(5):
            mdl = LogisticRegression(C=1e4, max_iter=300).fit(X[f != k], y[f != k]); pr = np.clip(mdl.predict_proba(X[f == k])[:, 1], 1e-6, 1 - 1e-6); ll[f == k, e_] = -(y[f == k] * np.log(pr) + (1 - y[f == k]) * np.log(1 - pr))
    ids = pd.factorize(g.w)[0]; d = np.bincount(ids, ll[:, 0] - ll[:, 1]); nc = np.bincount(ids); rr = np.random.default_rng(0); Rm = rr.multinomial(len(d), np.full(len(d), 1 / len(d)), size=2000); s = ((Rm @ d) / (Rm @ nc)).std(); m_ = d.sum() / nc.sum()
    print(f"{name}: {len(ws)} windows, {len(g)} cells | log loss base {ll[:, 0].mean():.5f} ext {ll[:, 1].mean():.5f} | improvement {m_:+.6f} (SE {s:.6f}, z {m_ / s:+.2f})", flush=True)


test(D[D.inst == "2031-03"], "March dev"); test(D[D.inst.isin(MONTHS)], "January + February")
for mo in MONTHS:
    test(D[D.inst == mo], mo)
