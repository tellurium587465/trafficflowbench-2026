"""G181 (11/01, ledgers #134-#137, strategy T2_DEEP_REVIEW 15): the middle of the parts whose sign splits by month, decode only, final A's flows.
Ongoing, M3 dev 356 windows and January, February (SCG path, q25_nomem OOF; X1 arms on the fold-2 networks of G135, base against x1, 4 seed mean):
  #135 soft far gate: in place of the hard gate, the links outside the zone and with no evidence get x f before best_set: s25, s50 (D > 3), dec (d = 2 x0.75, 3 x0.5, >= 4 x0.25)
  #136 half X1: hf = mean of base and X1 probabilities, b25, b50 = base + beta (X1 - base) in logit; read on A's flow and on the raw decode (B's April flow)
  #134 Platt per horizon row: a_h logit p + b_h fitted on January + February OOF, applied before best_set (on those months: fitted on the other month)
Onset, M3 dev 100 windows and January, February (G170 path):
  #137 B by earlier days: bk1 = 0.25 ** (k / 3), bk2 = {1: 0.6, 2: 0.4, >= 3: 0.25}; plt = #134 on T+30
Identity first: A against g132's base, SCG and PE against g119, onset base against g108. Hold-out days are not read.
usage: TFB_M3VER=v3 py -B -X utf8 g181_decode_arms.py"""
import sys, pickle, glob, importlib.util
R = r"C:/Users/user/tfb/review/"; sys.dont_write_bytecode = True
src = open(R + "g119_month_postproc.py", encoding="utf-8").read().split('if __name__ == "__main__":')[0]
sys.argv = [sys.argv[0], "run"]; exec(compile(src, "g119", "exec"), globals())
md = lambda nm, pa: (lambda sp: (lambda m: (sp.loader.exec_module(m), m)[1])(importlib.util.module_from_spec(sp)))(importlib.util.spec_from_file_location(nm, pa))
g = md("g121", R + "g121_ongoing_oracle.py"); g132 = md("g132", R + "g132_p3_grid.py"); g2 = md("g120", R + "g120_onset_gate.py")
from sklearn.linear_model import LogisticRegression
V, best_set = g.V, g.best_set; lg = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6))); sg = lambda x: 1 / (1 + np.exp(-x))
OOF = [pickle.load(open(W + f"oof_q25_nomem_s{s}.pkl", "rb")) for s in (3, 4)]


def fit_platt(ks):
    out = {}
    for c_, rows_ in (("queue_ongoing", range(6)), ("queue_onset", (5,))):
        for h in rows_:
            x, y = [], []
            for k in ks:
                if str(T.META["cond"][k]) == c_:
                    lab, el = T.LAB[k]; e = el[h].astype(bool); x.append(lg((OOF[0][k] + OOF[1][k])[h][e] / 2)); y.append(lab[h][e].astype(int))
            x, y = np.concatenate(x), np.concatenate(y); m = LogisticRegression(C=1e6, max_iter=300).fit(x[:, None], y); out[(c_, h)] = (float(m.coef_[0, 0]), float(m.intercept_[0]))
    return out


def platt(p, c_, pl):
    q = p.copy()
    for h in (range(6) if c_ == "queue_ongoing" else (5,)):
        a_, b_ = pl[(c_, h)]; q[h] = sg(a_ * lg(p[h]) + b_)
    return q


ALL = np.flatnonzero(T.META["is_sel"] & np.isin(MON, MONTHS)); PL = {"all": fit_platt(ALL), "2031-01": fit_platt(ALL[MON[ALL] == "2031-02"]), "2031-02": fit_platt(ALL[MON[ALL] == "2031-01"])}; print("Platt, all, ongoing rows:", {h: tuple(round(x, 3) for x in PL["all"][("queue_ongoing", h)]) for h in range(6)}, "onset", tuple(round(x, 3) for x in PL["all"][("queue_onset", 5)]), flush=True)


def dist(lq, n):
    q = np.flatnonzero(lq); return np.full(n, 99) if not len(q) else np.abs(np.arange(n)[:, None] - q[None, :]).min(axis=1)


def soft(d, ex):
    return {"s25": np.where((d > 3) & ~ex, 0.25, 1.0), "s50": np.where((d > 3) & ~ex, 0.5, 1.0), "dec": np.where(ex, 1.0, np.select([d <= 1, d == 2, d == 3], [1.0, 0.75, 0.5], 0.25))}


XA = g.X1; PTM = [pickle.load(open(W + f"probs_m3v3__tfm4_s{s}.pkl", "rb")) for s in range(5)]; PB = [pickle.load(open(W + f"probs_m3v3__c24f_attnglob_s{s}.pkl", "rb")) for s in (0, 1)]; rows = []
CC = ("A", "s25", "s50", "dec", "hf", "b25", "b50", "plt", "rx0", "rx1", "rhf", "rb25", "rb50")
mixl = lambda x0, x1, b: sg(lg(x0) + b * (lg(x1) - lg(x0)))
for w in g.DEV:
    m = G.META[w]; i = G.POS[w]; Xw = np.asarray(G.X[G.OFF[i]:G.OFF[i + 1]]); pl = g.FS.lgb_ongoing(Xw); pt = sum(d_[w] for d_ in PTM) / 5; x0 = (PB[0][w] + PB[1][w]) / 2; lq = np.asarray(Xw[:, G.CI], float) == 1; P, _, q = m3v2._panel(m["panel"]); t = m["t"]; d0 = t - (t % 288) + 60; _, cone, _ = g.g55.parts(w); nr = g132.nears(lq, P.n)[3]; dd = dist(lq, P.n); sc = dict.fromkeys(CC, 0.0)
    for k, pr in m3v2.kdist(m["day"]).items():
        hid = V._hidden_today(m, k); rr = [r for r in range(d0, t) if r not in hid]; ev_ = q[rr].any(axis=0) if rr else np.zeros(P.n, bool); s2, _ = m3v2.seen(m["panel"], t, k, -18, 12); ex = G.dil(ev_ | s2, 2); s, nd = m3v2.seen(m["panel"], t, k, -18, 12); sw = np.where(G.dil(s, 1), 1.0, 0.85)[None, :] if nd else 1.0; cf = np.where(ex[None, :], 1.0, np.where(cone < 0.999, 0.5, 1.0)); x1 = XA[(w, k)]
        bs = lambda pb: best_set(pb.ravel(), np.ones(pb.size, bool)).reshape(pb.shape); fl = lambda x: (0.8 * (0.8 * x + 0.2 * pt) + 0.2 * pl) * cf * sw; pA = fl(x1); fs = soft(dd, ex); S_ = dict(A=g132.gate(bs(pA), ex, nr), plt=g132.gate(bs(platt(pA, "queue_ongoing", PL["all"])), ex, nr), rx0=bs(x0), rx1=bs(x1))
        for nm, f_ in fs.items():
            S_[nm] = bs(pA * f_[None, :])
        for nm, x_ in (("hf", (x0 + x1) / 2), ("b25", mixl(x0, x1, 0.25)), ("b50", mixl(x0, x1, 0.5))):
            S_[nm] = g132.gate(bs(fl(x_)), ex, nr); S_["r" + nm] = bs(x_)
        for c_ in CC:
            sc[c_] += pr * G.iou(S_[c_], m)
    rows.append(dict(w=w, cond=m["cond"], panel=m["panel"], day=m["day"], hour=(t % 288) // 12, **sc))
Dm = pd.DataFrame(rows); ref = pd.read_csv(W + "p3_grid/g132_grid_m3.csv").set_index("w")["l20_t20_c50_D3"]; print(f"M3 ongoing {len(Dm)} | identity A vs g132 {np.abs(ref.loc[Dm.w].to_numpy() - Dm.A.to_numpy()).max():.2e}", flush=True); Dm.to_csv(W + "g181_m3_og.csv", index=False)
X1M = [pickle.load(open(W + f"x1_months/capB0_s{s}_fold2030-12.pkl", "rb")) for s in range(4)]; rows = []
for mo in MONTHS:
    ks = np.flatnonzero(T.META["is_sel"] & (MON == mo) & (T.META["cond"] == "queue_ongoing"))
    for p in sorted({str(T.META["panel"][k]) for k in ks}):
        P = load(p, "train"); key = (mo, p, "mask"); CTX.clear(); sim_blank.cache_clear(); CTX[key] = Ctx(key, P, np.where(P.blank, np.nan, P.V), P.V); c_ = CTX[key]
        for k in [k for k in ks if str(T.META["panel"][k]) == p]:
            t = int(T.META["t"][k]); lab, el = T.LAB[k]; y, e = lab.astype(bool), el.astype(bool); Xw = np.asarray(XN[OFFN[k]:OFFN[k + 1]]); lq = np.asarray(Xw[:, G.CI], float) == 1; p_ = (OOF[0][k] + OOF[1][k]) / 2; dom = int(P.times[t][8:10]) - 1; cone = cone_parts(c_, t, lq, P.length); nr = g132.nears(lq, P.n)[3]; dd = dist(lq, P.n); r = dict(w=int(k), month=mo, panel=p, day=dom); hx = all(k in x_ for x_ in X1M)
            if hx:
                x0 = sum(x_[k][0] for x_ in X1M) / 4; x1 = sum(x_[k][1] for x_ in X1M) / 4
            for kk, pr in m3v2.kdist(dom, cap=3).items():
                ev_ = today_ev(c_, t, kk); s2, _ = seen(c_, t, kk, -18, 12); ex = G.dil(ev_ | s2, 2); s, nd = seen(c_, t, kk, -18, 12); sw = np.where(G.dil(s, 1), 1.0, 0.85)[None, :] if nd else 1.0; cf = np.where(ex[None, :], 1.0, np.where(cone < 0.999, 0.5, 1.0)); pb = p_ * cf * sw; fs = soft(dd, ex); sc = dict(SCG=g132.gate(decode(pb), ex, nr), nogate=decode(pb), plt=g132.gate(decode(platt(pb, "queue_ongoing", PL[mo])), ex, nr))
                for nm, f_ in fs.items():
                    sc[nm] = decode(pb * f_[None, :])
                if hx:
                    fx = lambda x_: x_ * cf * sw; sc.update(X0=g132.gate(decode(fx(x0)), ex, nr), X1=g132.gate(decode(fx(x1)), ex, nr), hf=g132.gate(decode(fx((x0 + x1) / 2)), ex, nr), b25=g132.gate(decode(fx(mixl(x0, x1, 0.25))), ex, nr), b50=g132.gate(decode(fx(mixl(x0, x1, 0.5))), ex, nr), rx0=decode(x0), rx1=decode(x1), rhf=decode((x0 + x1) / 2), rb25=decode(mixl(x0, x1, 0.25)), rb50=decode(mixl(x0, x1, 0.5)))
                for nm, s_ in sc.items():
                    r[nm] = r.get(nm, 0.0) + pr * iou(s_, y, e, "queue_ongoing")
            rows.append(r)
        load.cache_clear()
Dt = pd.DataFrame(rows); Dt.to_csv(W + "g181_train_og.csv", index=False); ref = pd.read_csv(W + "g119_month_postproc.csv", dtype={"w": str}); rs = ref[(ref.vis == "mask") & (ref.cond == "queue_ongoing")].set_index(["month", "w"]).SCG; print("identity SCG vs g119", float(max(abs(rs.loc[(r.month, str(r.w))] - r.SCG) for r in Dt.itertuples())), len(Dt), "with X1", int(Dt.X1.notna().sum()), flush=True)
sv = g2.__dict__; PN = g2.nn_blend(); phys = lgb.Booster(model_file=W + "physon_base.txt"); ev = PO.month_events("validation", PO.rarity_table()); ONV = ("base", "bk1", "bk2", "plt")


def on_sets(c, p_, ph, a, t, k, pl_):
    s, nd = seen(c, t, k, -6, 12); evd = G.dil(s, 2); pa = p_.copy(); pa[5] = (1 - a) * p_[5] + a * ph
    def rule(B):
        x = pa.copy()
        if nd:
            x[-1] = x[-1] * np.where(evd, 1.0, B)
        return x
    nk = min(k, 3); return dict(base=decode(rule(0.25))[-1], bk1=decode(rule(0.25 ** (nk / 3) if nk else 1.0))[-1], bk2=decode(rule({0: 1.0, 1: 0.6, 2: 0.4, 3: 0.25}[nk]))[-1], plt=decode(platt(rule(0.25), "queue_onset", pl_))[-1])


def i1(s_, y, e):
    u = ((s_ | y) & e).sum(); return 1.0 if u == 0 else float((s_ & y & e).sum() / u)


rows = []
for w in g2.DEV:
    m = G.META[w]; key = ("val", m["panel"]); Pv = load(m["panel"], "validation")
    if key not in CTX:
        CTX[key] = Ctx(key, Pv, Pv.V)
    i = G.POS[w]; Xw = np.asarray(G.X[G.OFF[i]:G.OFF[i + 1]]); ph = phys.predict(PO.feats(Xw)); a = PO.adaptive_a(ev, m["day"]); y, e = m["y"].T.astype(bool)[-1], m["m"].T.astype(bool)[-1]; sc = dict.fromkeys(ONV, 0.0)
    for k, pr in m3v2.kdist(m["day"], cap=3).items():
        S_ = on_sets(CTX[key], PN[w], ph, a, m["t"], k, PL["all"])
        for v in ONV:
            sc[v] += pr * i1(S_[v], y, e)
    rows.append(dict(w=w, cond=m["cond"], panel=m["panel"], day=m["day"], hour=(m["t"] % 288) // 12, **sc))
Dn = pd.DataFrame(rows); o = pd.read_csv(W + "g108_rectangles.csv").set_index("w").A.reindex(Dn.w).to_numpy(); print(f"M3 onset {len(Dn)} | identity {np.abs(o - Dn.base.to_numpy()).max():.2e}", flush=True); Dn.to_csv(W + "g181_m3_on.csv", index=False)
FF = phys_feats(); rows = []
for mo in MONTHS:
    mdl = phys_lomo(FF, mo); F_m = rarity_lomo(mo); ks = np.flatnonzero(T.META["is_sel"] & (MON == mo) & (T.META["cond"] == "queue_onset"))
    for p in sorted({str(T.META["panel"][k]) for k in ks}):
        P = load(p, "train"); key = (mo, p, "mask"); CTX.clear(); sim_blank.cache_clear(); CTX[key] = Ctx(key, P, np.where(P.blank, np.nan, P.V), P.V); c = CTX[key]; evm = month_events(c, p, P, mo, F_m)
        for k in ks[np.array([str(T.META["panel"][k]) == p for k in ks])]:
            t = int(T.META["t"][k]); dom = int(P.times[t][8:10]) - 1; lab, el = T.LAB[k]; Xw = np.asarray(XN[OFFN[k]:OFFN[k + 1]]); y, e = lab[5].astype(bool), el[5].astype(bool); sc = dict.fromkeys(ONV, 0.0)
            for kk, pr in m3v2.kdist(dom, cap=3).items():
                S_ = on_sets(c, (OOF[0][k] + OOF[1][k]) / 2, mdl.predict(PO.feats(Xw)), PO.adaptive_a(evm, dom), t, kk, PL[mo])
                for v in ONV:
                    sc[v] += pr * i1(S_[v], y, e)
            rows.append(dict(w=int(k), month=mo, panel=p, day=dom, **sc))
        load.cache_clear()
Dto = pd.DataFrame(rows); Dto.to_csv(W + "g181_train_on.csv", index=False)


def clse(d, cl, n_boot=2000):
    ids = pd.factorize(cl)[0]; M = np.bincount(ids, d); N = np.bincount(ids); Rm = np.random.default_rng(0).multinomial(len(M), np.full(len(M), 1 / len(M)), size=n_boot); return float(((Rm @ M) / (Rm @ N)).std())


def line(nm, Dmm, a, b, Dtt, ta, tb):
    s = m3v2.summary(Dmm, a, b, "private"); sm = m3v2.summary(Dmm, a, b, "validation", n_boot=500, n_draw=200); out = [f"{nm:22s} M3 {s['t2_weighted']:+.4f} (SE {s['se']:.4f}, z {s['t2_weighted'] / max(s['se'], 1e-9):+.2f}) Mar comp {sm['t2_weighted']:+.4f}"]
    for mo in MONTHS:
        d = Dtt[(Dtt.month == mo)].dropna(subset=[ta, tb]); x = (d[tb] - d[ta]).to_numpy(); out.append(f"{mo[-2:]} {x.mean():+.4f} ({clse(x, d.panel + '_' + d.day.astype(str)):.4f})")
    print(" | ".join(out), flush=True)


print("--- ongoing, against the hard gate"); line("no gate", Dm.assign(ng=Dm.s50 * 0 + Dm.A), "A", "ng", Dt, "SCG", "nogate")
for v in ("s25", "s50", "dec"):
    line(f"#135 {v}", Dm, "A", v, Dt, "SCG", v)
print("--- ongoing, X1: A flow against full X1, and raw against base"); line("X1 full (raw x0->x1)", Dm, "rx0", "rx1", Dt, "rx0", "rx1")
for v in ("hf", "b25", "b50"):
    line(f"#136 {v} vs full, A flow", Dm, "A", v, Dt, "X1", v); line(f"#136 {v} vs base, raw", Dm, "rx0", "r" + v, Dt, "rx0", "r" + v)
line("#134 Platt ongoing", Dm, "A", "plt", Dt, "SCG", "plt"); print("--- onset, against uniform 0.25")
for v in ("bk1", "bk2", "plt"):
    line(f"#137/#134 {v}", Dn, "base", v, Dto, "base", v)
