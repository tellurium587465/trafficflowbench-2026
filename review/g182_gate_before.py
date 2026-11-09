"""G182 (11/01, ledger #146, strategy T2_DEEP_REVIEW 18): the far gate drops positives after best_set has chosen k, so the set left is not the expected-IoU best. Here the cells the gate drops get p = 0 and best_set runs again, then the gate again (it keeps the contiguity rule). M3 dev ongoing (A flow) and January, February (SCG path).
Identity: the current columns against g132's base and g119's SCG. Hold-out days are not read.
usage: TFB_M3VER=v3 py -B -X utf8 g182_gate_before.py"""
import sys, pickle, importlib.util
R = r"C:/Users/user/tfb/review/"; sys.dont_write_bytecode = True
src = open(R + "g119_month_postproc.py", encoding="utf-8").read().split('if __name__ == "__main__":')[0]
sys.argv = [sys.argv[0], "run"]; exec(compile(src, "g119", "exec"), globals())
md = lambda nm, pa: (lambda sp: (lambda m: (sp.loader.exec_module(m), m)[1])(importlib.util.module_from_spec(sp)))(importlib.util.spec_from_file_location(nm, pa))
g = md("g121", R + "g121_ongoing_oracle.py"); g132 = md("g132", R + "g132_p3_grid.py"); V, best_set = g.V, g.best_set; OOF = [pickle.load(open(W + f"oof_q25_nomem_s{s}.pkl", "rb")) for s in (3, 4)]; PTM = [pickle.load(open(W + f"probs_m3v3__tfm4_s{s}.pkl", "rb")) for s in range(5)]


def two(pb, bs, ex, nr):
    s0 = bs(pb); s1 = g132.gate(s0, ex, nr); gone = s0 & ~s1; return s1, g132.gate(bs(np.where(gone, 0.0, pb)), ex, nr)


rows = []
for w in g.DEV:
    m = G.META[w]; i = G.POS[w]; Xw = np.asarray(G.X[G.OFF[i]:G.OFF[i + 1]]); pl = g.FS.lgb_ongoing(Xw); pt = sum(d_[w] for d_ in PTM) / 5; lq = np.asarray(Xw[:, G.CI], float) == 1; P, _, q = m3v2._panel(m["panel"]); t = m["t"]; d0 = t - (t % 288) + 60; _, cone, _ = g.g55.parts(w); nr = g132.nears(lq, P.n)[3]; a = b = 0.0
    for k, pr in m3v2.kdist(m["day"]).items():
        hid = V._hidden_today(m, k); rr = [r for r in range(d0, t) if r not in hid]; ev_ = q[rr].any(axis=0) if rr else np.zeros(P.n, bool); s2, _ = m3v2.seen(m["panel"], t, k, -18, 12); ex = G.dil(ev_ | s2, 2); s, nd = m3v2.seen(m["panel"], t, k, -18, 12); sw = np.where(G.dil(s, 1), 1.0, 0.85)[None, :] if nd else 1.0; cf = np.where(ex[None, :], 1.0, np.where(cone < 0.999, 0.5, 1.0)); pb = (0.8 * (0.8 * g.X1[(w, k)] + 0.2 * pt) + 0.2 * pl) * cf * sw
        x, y = two(pb, lambda p_: best_set(p_.ravel(), np.ones(p_.size, bool)).reshape(p_.shape), ex, nr); a += pr * G.iou(x, m); b += pr * G.iou(y, m)
    rows.append(dict(w=w, cond=m["cond"], panel=m["panel"], day=m["day"], hour=(t % 288) // 12, A=a, B=b))
Dm = pd.DataFrame(rows); ref = pd.read_csv(W + "p3_grid/g132_grid_m3.csv").set_index("w")["l20_t20_c50_D3"]; print(f"identity {np.abs(ref.loc[Dm.w].to_numpy() - Dm.A.to_numpy()).max():.2e}, changed {(Dm.A != Dm.B).sum()}", flush=True); s = m3v2.summary(Dm, "A", "B", "private"); print(f"M3 dev: April composition {s['t2_weighted']:+.4f} (SE {s['se']:.4f}, z {s['t2_weighted'] / max(s['se'], 1e-9):+.2f}), raw {(Dm.B - Dm.A).mean():+.4f}", flush=True)
rows = []
for mo in MONTHS:
    ks = np.flatnonzero(T.META["is_sel"] & (MON == mo) & (T.META["cond"] == "queue_ongoing"))
    for p in sorted({str(T.META["panel"][k]) for k in ks}):
        P = load(p, "train"); key = (mo, p, "mask"); CTX.clear(); sim_blank.cache_clear(); CTX[key] = Ctx(key, P, np.where(P.blank, np.nan, P.V), P.V); c_ = CTX[key]
        for k in [k for k in ks if str(T.META["panel"][k]) == p]:
            t = int(T.META["t"][k]); lab, el = T.LAB[k]; y_, e = lab.astype(bool), el.astype(bool); Xw = np.asarray(XN[OFFN[k]:OFFN[k + 1]]); lq = np.asarray(Xw[:, G.CI], float) == 1; p_ = (OOF[0][k] + OOF[1][k]) / 2; dom = int(P.times[t][8:10]) - 1; cone = cone_parts(c_, t, lq, P.length); nr = g132.nears(lq, P.n)[3]; a = b = 0.0
            for kk, pr in m3v2.kdist(dom, cap=3).items():
                ev_ = today_ev(c_, t, kk); s2, _ = seen(c_, t, kk, -18, 12); ex = G.dil(ev_ | s2, 2); s, nd = seen(c_, t, kk, -18, 12); sw = np.where(G.dil(s, 1), 1.0, 0.85)[None, :] if nd else 1.0; cf = np.where(ex[None, :], 1.0, np.where(cone < 0.999, 0.5, 1.0)); x, y = two(p_ * cf * sw, decode, ex, nr); a += pr * iou(x, y_, e, "queue_ongoing"); b += pr * iou(y, y_, e, "queue_ongoing")
            rows.append(dict(w=int(k), month=mo, panel=p, day=dom, A=a, B=b))
        load.cache_clear()
Dt = pd.DataFrame(rows); rs = pd.read_csv(W + "g119_month_postproc.csv", dtype={"w": str}); rs = rs[(rs.vis == "mask") & (rs.cond == "queue_ongoing")].set_index(["month", "w"]).SCG; print("identity SCG", float(max(abs(rs.loc[(r.month, str(r.w))] - r.A) for r in Dt.itertuples())), flush=True)
for mo, d in Dt.groupby("month"):
    x = (d.B - d.A).to_numpy(); ids = pd.factorize(d.panel + "_" + d.day.astype(str))[0]; M = np.bincount(ids, x); N = np.bincount(ids); Rm = np.random.default_rng(0).multinomial(len(M), np.full(len(M), 1 / len(M)), size=2000); print(mo, len(d), f"{x.mean():+.4f} (SE {((Rm @ M) / (Rm @ N)).std():.4f}) changed {(x != 0).sum()}", flush=True)
