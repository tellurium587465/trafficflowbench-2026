"""G274 (11/04, ledger #227 arm, declared by the strategy before the results): the far gate keeps the band 4-8 links upstream of each
tail of the last queue (tails = the upstream ends of lq's runs; links ordered upstream -> downstream): those links are added to the
gate's "near" set, nothing else changes.
  jf : the 316 January / February selector ongoing windows, G229's flow (capB0 s0-3, G119's SCG flow with the row T), G119's far_gate
       patched; paired, cluster bootstrap (month x panel x day)
  dev: M3 v3 dev 356 windows, A12's flow of G212 (A13 on dev) with g132.gate's near set widened the same way; even days (judge, April
       composition), all and the top coverage third; odd days as a record
usage: TFB_X3=x3af TFB_M3VER=v3 py -B -X utf8 g274_far_gate_band_arm.py jf | dev"""
import os, sys, pickle, importlib.util
os.environ.setdefault("TFB_M3VER", "v3")
import numpy as np, pandas as pd
R = r"C:/Users/user/tfb/"; MODE = sys.argv[1]


def band(lq):
    b = np.zeros(len(lq), bool)
    for j in range(len(lq)):
        if lq[j] and (j == 0 or not lq[j - 1]):
            b[max(0, j - 8):max(0, j - 3)] = True                # links j-8 .. j-4
    return b


if MODE == "jf":
    a_ = sys.argv; sys.argv = ["g119", "run"]; spec = importlib.util.spec_from_file_location("g119", R + "review/g119_month_postproc.py"); g = importlib.util.module_from_spec(spec); spec.loader.exec_module(g); sys.argv = a_
    src = open(R + "review/g119_month_postproc.py", encoding="utf-8").read(); i0 = src.index("def far_gate(c, sel, t, k, lq):"); i1 = src.index("def ongoing_pred(")
    fsrc = src[i0:i1].replace("def far_gate(c, sel, t, k, lq):", "def far_gate_band(c, sel, t, k, lq):").replace("        near[max(0, j - 3):j + 4] = True\n", "        near[max(0, j - 3):j + 4] = True\n    near |= BAND_(lq)\n")
    assert fsrc.count("BAND_(lq)") == 1; g.BAND_ = band; exec(compile(fsrc, "fgb", "exec"), g.__dict__); fg0 = g.far_gate
    T, MON = g.T, g.MON; PR = [pickle.load(open(g.W + f"probs_jfmh0__capB0_s{s}_fold2030-12.pkl", "rb")) for s in range(4)]; rows = []
    KS = [k for k in np.flatnonzero(T.META["is_sel"] & np.isin(MON, g.MONTHS)) if str(T.META["cond"][k]) == "queue_ongoing"]
    for mo in g.MONTHS:
        for p in sorted({str(T.META["panel"][k]) for k in KS if MON[k] == mo}):
            P = g.load(p, "train"); key = (mo, p, "mask"); g.CTX.clear(); g.sim_blank.cache_clear(); g.CTX[key] = g.Ctx(key, P, np.where(P.blank, np.nan, P.V), P.V); c = g.CTX[key]
            for k in [k for k in KS if MON[k] == mo and str(T.META["panel"][k]) == p]:
                t = int(T.META["t"][k]); dom = int(P.times[t][8:10]) - 1; lab, el = T.LAB[k]; y, e = lab.astype(bool), el.astype(bool)
                lq = np.asarray(g.XN[g.OFFN[k]:g.OFFN[k + 1]][:, g.G.CI], float) == 1; ot = c.obs[t].astype(bool); lq2 = np.where(ot, c.q[t].astype(bool), lq)
                p_ = sum(d[str(k)] for d in PR) / len(PR); cone = g.cone_parts(c, t, lq2, P.length); r = dict(w=int(k), month=mo, panel=p, day=dom)
                for nm, fg in (("base", fg0), ("band", g.far_gate_band)):
                    g.far_gate = fg; r[nm] = sum(pr * g.iou(g.ongoing_pred(c, p_ * g.cone_f(c, cone, t, kk), t, kk, True, lq2), y, e, "queue_ongoing") for kk, pr in g.m3v2.kdist(dom, cap=3).items())
                g.far_gate = fg0; rows.append(r)
            g.load.cache_clear()
    D = pd.DataFrame(rows); D.to_csv(g.W + "g274_jf.csv", index=False); ref = pd.read_csv(g.W + "g229_jf.csv").set_index("w").trow.reindex(D.w).to_numpy()
    cl = (D.month.astype(str) + "_" + D.panel + "_" + D.day.astype(str)).to_numpy(); d = (D.band - D.base).to_numpy()
    ids = pd.factorize(cl)[0]; M_ = np.bincount(ids, d); N_ = np.bincount(ids); Rm = np.random.default_rng(0).multinomial(len(M_), np.full(len(M_), 1 / len(M_)), size=2000); se = float(((Rm @ M_) / (Rm @ N_)).std())
    print(f"identity base vs G229 trow max |d| {np.abs(ref - D.base.to_numpy()).max():.2e} | Jan/Feb band - base {d.mean():+.4f} (SE {se:.4f}, z {d.mean() / max(se, 1e-12):+.2f}) | Jan {d[(D.month == '2031-01').to_numpy()].mean():+.4f} Feb {d[(D.month == '2031-02').to_numpy()].mean():+.4f} | windows changed {int((np.abs(d) > 1e-12).sum())}", flush=True)
    sys.exit()
assert MODE == "dev"
R_ = R + "review/"; END = 'exec(compile(s, "g175a", "exec"))'; src = open(R_ + "g212_a10_read.py", encoding="utf-8").read(); assert src.count(END) == 1
exec(compile(src.replace(END, ""), "g212", "exec"))
FA = '(0.8 * (0.8 * ((1 - X3) * (0.8 * xs + 0.2 * pt) + X3 * px) + 0.2 * pu) + 0.2 * pl) * cf * sw'
sub('sc["a12"] += pr * G.iou(dec(tmp(' + FA + ')), m)',
    'sc["a12"] += pr * G.iou(dec(tmp(' + FA + ')), m); pb_ = tmp(' + FA + '); sc["a12fb"] += pr * G.iou(g132.gate(best_set(pb_.ravel(), np.ones(pb_.size, bool)).reshape(pb_.shape), ex, nr | BAND_(lq)), m)')
sub('"a12", "b10"); cols', '"a12", "b10", "a12fb"); cols')
sub('D.to_csv(W + "g212_dev.csv"', 'D.to_csv(W + "g274_dev.csv"')
sub('    Df = train_B(); Df.to_csv(W + "g212_train.csv", index=False)\n',
    '    cov_ = COV(); D["cov"] = D.w.map(cov_); odd = (D.day % 2 == 0).to_numpy(); E_ = D[~odd].reset_index(drop=True); top = E_[E_["cov"] > D["cov"].quantile(2 / 3)].reset_index(drop=True)\n'
    '    for nm_, d_ in (("even days all", E_), ("even days top third", top), ("odd days all (record)", D[odd].reset_index(drop=True))):\n'
    '        s_ = m3v2.summary(d_, "a12", "a12fb", "private", n_boot=2000, n_draw=500); print(f"M3 band - base, {nm_} ({len(d_)}): {s_[\'t2_weighted\']:+.4f} (SE {s_[\'se\']:.4f}, z {s_[\'t2_weighted\'] / max(s_[\'se\'], 1e-12):+.2f}) | windows changed {int((~np.isclose(d_.a12fb, d_.a12)).sum())}", flush=True)\n'
    '    sys.exit()\n')
BAND_ = band
exec(compile(s, "g175g", "exec"))
