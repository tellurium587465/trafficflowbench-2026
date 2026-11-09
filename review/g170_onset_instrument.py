"""G170 (11/01, ledgers #120, #122; strategy T2_DEEP_REVIEW 8-1 A1, 8-2 B2). Does M3's way of choosing and scoring onset windows change the comparison of the onset components?
Final A's onset flow, decode only. Variants: B 0.25 (base), 0.5, 0.7, 1.0; SKIP (no rule when no link has evidence); physics a = 0, 0.3 (base adaptive); temperature 0.8, 1.2 on the base;
place = B 0.25 with the window's T+30 total restored; amt 0.5, 0.7 = no rule, a uniform factor where the rule would act.
January and February, q25_nomem seeds 3, 4 OOF, month left out as G119, three readings of the same decoded sets at T+30:
  L  (the instrument of G119): T.LAB, eligible cells
  F  (i) all finite cells of the training truth, all windows
  M  (ii) M3's way: the horizon is seen through March's observation pattern of the same corridor and clock (8 draws of dev days), the window counts only if a queue is seen at T+30, only the seen eligible cells are scored
March dev (100 windows, hold-out days not read): the same variants under M3 itself, for #122 and for judge. Identity: base against g108 A and g119 PE.
usage: TFB_M3VER=v3 py -B -X utf8 g170_onset_instrument.py [time | run]"""
import sys, pickle
R = r"C:/Users/user/tfb/review/"; sys.dont_write_bytecode = True; RUN = (sys.argv[1:] or ["run"])[0]
src = open(R + "g119_month_postproc.py", encoding="utf-8").read().split('if __name__ == "__main__":')[0]
sys.argv = [sys.argv[0], "run"]; exec(compile(src, "g119", "exec"), globals())
import importlib.util
sp = importlib.util.spec_from_file_location("g120", R + "g120_onset_gate.py"); g2 = importlib.util.module_from_spec(sp); sp.loader.exec_module(g2)
VARS = ("base", "B0.5", "B0.7", "B1.0", "skip", "a0", "a0.3", "T0.8", "T1.2", "place", "amt0.5", "amt0.7"); ND = 8
lg = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6))); sg = lambda x: 1 / (1 + np.exp(-x))


def sets(c, p_, ph, a, t, k):
    s, nd = seen(c, t, k, -6, 12); ev = G.dil(s, 2)

    def mix(a_):
        q = p_.copy(); q[5] = (1 - a_) * p_[5] + a_ * ph; return q

    def rule(pb, B):
        x = pb.copy()
        if nd:
            x[-1] = x[-1] * np.where(ev, 1.0, B)
        return x
    pa = mix(a); b = rule(pa, 0.25); pl = b.copy()
    if nd:
        pl[-1] = np.minimum(pl[-1] * pa[-1].sum() / max(pl[-1].sum(), 1e-9), 1 - 1e-6)
    am = lambda c_: (lambda x: (x.__setitem__(-1, x[-1] * c_ if nd else x[-1]), x)[1])(pa.copy())
    pm = {"base": b, "B0.5": rule(pa, 0.5), "B0.7": rule(pa, 0.7), "B1.0": pa, "skip": b if (nd and s.any()) else pa, "a0": rule(mix(0.0), 0.25), "a0.3": rule(mix(0.3), 0.25),
          "T0.8": sg(lg(b) / 0.8), "T1.2": sg(lg(b) / 1.2), "place": pl, "amt0.5": am(0.5), "amt0.7": am(0.7)}
    return {v: decode(x)[-1] for v, x in pm.items()}


def iou1(s_, y, e):
    u = ((s_ | y) & e).sum(); return 1.0 if u == 0 else float((s_ & y & e).sum() / u)


def march(limit=None):
    PN = g2.nn_blend(); phys = lgb.Booster(model_file=W + "physon_base.txt"); ev = PO.month_events("validation", PO.rarity_table()); rows = []
    for w in g2.DEV[:limit]:
        m = G.META[w]; key = ("val", m["panel"]); P = load(m["panel"], "validation")
        if key not in CTX:
            CTX[key] = Ctx(key, P, P.V)
        i = G.POS[w]; Xw = np.asarray(G.X[G.OFF[i]:G.OFF[i + 1]]); ph = phys.predict(PO.feats(Xw)); a = PO.adaptive_a(ev, m["day"]); y, e = m["y"].T.astype(bool)[-1], m["m"].T.astype(bool)[-1]; sc = dict.fromkeys(VARS, 0.0)
        for k, pr in m3v2.kdist(m["day"], cap=3).items():
            S = sets(CTX[key], PN[w], ph, a, m["t"], k)
            for v in VARS:
                sc[v] += pr * iou1(S[v], y, e)
        rows.append(dict(w=w, cond=m["cond"], panel=m["panel"], day=m["day"], hour=(m["t"] % 288) // 12, **sc))
    return pd.DataFrame(rows)


def march_vis(p):
    Pm = load(p, "validation"); dev = [d for d in range(Pm.T // 288) if not m3v2.is_holdout(np.array([d]))[0]]; return np.isfinite(Pm.V) & np.isfinite(Pm.Q), dev


def train_months(limit=None):
    OOF = [pickle.load(open(W + f"oof_q25_nomem_s{s}.pkl", "rb")) for s in (3, 4)]; FF = phys_feats() if limit is None else None; rows = []
    for mo in MONTHS:
        mdl = phys_lomo(FF, mo) if FF is not None else lgb.Booster(model_file=W + "physon_base.txt"); F_m = rarity_lomo(mo); ks = np.flatnonzero(T.META["is_sel"] & (MON == mo) & (T.META["cond"] == "queue_onset"))
        for p in sorted({str(T.META["panel"][k]) for k in ks}):
            Mv, dev = march_vis(p); P = load(p, "train"); key = (mo, p, "mask"); CTX.clear(); sim_blank.cache_clear(); CTX[key] = Ctx(key, P, np.where(P.blank, np.nan, P.V), P.V); c = CTX[key]; ev = month_events(c, p, P, mo, F_m)
            for k in ks[np.array([str(T.META["panel"][k]) == p for k in ks])]:
                t = int(T.META["t"][k]); dom = int(P.times[t][8:10]) - 1; lab, el = T.LAB[k]; Xw = np.asarray(XN[OFFN[k]:OFFN[k + 1]]); p_ = (OOF[0][k] + OOF[1][k]) / 2; ph = mdl.predict(PO.feats(Xw)); a = PO.adaptive_a(ev, dom)
                yL, eL = lab[5].astype(bool), el[5].astype(bool); r_ = t + 6; Vt = P.V[r_]; eF = np.isfinite(Vt); yF = eF & (np.nan_to_num(Vt, nan=1e9) <= P.vcut); rng = np.random.default_rng(k); ds = rng.choice(dev, ND, replace=False); eM = [eL & Mv[d * 288 + r_ % 288] for d in ds]; yM = [y_ & e_ for y_, e_ in ((yL, e_) for e_ in eM)]
                sc = {f"{l}_{v}": 0.0 for l in ("L", "F") for v in VARS}; mm = np.zeros((ND, len(VARS)))
                for kk, pr in m3v2.kdist(dom, cap=3).items():
                    S = sets(c, p_, ph, a, t, kk)
                    for iv, v in enumerate(VARS):
                        sc[f"L_{v}"] += pr * iou1(S[v], yL, eL); sc[f"F_{v}"] += pr * iou1(S[v], yF, eF)
                        for d_ in range(ND):
                            if yM[d_].any():
                                mm[d_, iv] += pr * iou1(S[v], yM[d_], eM[d_])
                sel = np.array([y_.any() for y_ in yM]); rows.append(dict(w=int(k), month=mo, panel=p, day=dom, nsel=int(sel.sum()), nY=int(yF.sum()), a=a, **sc, **{f"M_{v}": float(mm[sel, iv].sum()) for iv, v in enumerate(VARS)}))
                if limit and len(rows) >= limit:
                    return pd.DataFrame(rows)
            load.cache_clear()
    return pd.DataFrame(rows)


def ratio(D, nu, de, n_boot=2000):
    ids = pd.factorize(D.panel + "_" + D.day.astype(str))[0]; M = np.bincount(ids, nu); N = np.bincount(ids, de); Rm = np.random.default_rng(0).multinomial(len(M), np.full(len(M), 1 / len(M)), size=n_boot); return float(M.sum() / N.sum()), float(((Rm @ M) / (Rm @ N)).std())


if __name__ == "__main__":
    lim = 6 if RUN == "time" else None; t0 = TM(); Dm = march(lim); o = pd.read_csv(W + "g108_rectangles.csv").set_index("w").A.reindex(Dm.w).to_numpy()
    print(f"M3 dev {len(Dm)} windows {TM() - t0:.0f}s | identity against g108 A: {np.abs(o - Dm.base.to_numpy()).max():.2e}", flush=True); t0 = TM(); Dt = train_months(lim)
    ref = pd.read_csv(W + "g119_month_postproc.csv", dtype={"w": str}); ref = ref[ref.vis == "mask"].set_index(["month", "w"]).PE; ix = pd.MultiIndex.from_arrays([Dt.month, Dt.w.astype(str)]); dl = np.abs(ref.reindex(ix).to_numpy() - Dt.L_base.to_numpy())
    print(f"Jan, Feb {len(Dt)} windows {TM() - t0:.0f}s | identity L_base against g119 PE: {np.nanmax(dl):.2e}", flush=True)
    if RUN == "time":
        sys.exit()
    if np.abs(o - Dm.base.to_numpy()).max() > 1e-12 or np.nanmax(dl) > 1e-12:
        print("IDENTITY FAILED: stop"); sys.exit(1)
    Dm.to_csv(W + "g170_m3.csv", index=False); Dt.to_csv(W + "g170_train.csv", index=False)
    print(f"selected (window x draw) by M's rule: {Dt.nsel.sum() / (ND * len(Dt)):.3f}; by the number of true queue cells at T+30:", Dt.groupby(pd.cut(Dt.nY, [-1, 1, 2, 4, 8, 999])).nsel.mean().div(ND).round(3).to_dict())
    print("levels of the base: L", round(Dt.L_base.mean(), 4), "F", round(Dt.F_base.mean(), 4), "M", round(Dt.M_base.sum() / Dt.nsel.sum(), 4), "| March M3", round(Dm.base.mean(), 4))
    for mo, lab in (("pool", "Jan + Feb"), ("2031-01", "Jan"), ("2031-02", "Feb")):
        D = Dt if mo == "pool" else Dt[Dt.month == mo]; print(f"--- {lab}: variant minus base. L | F (i) all finite cells, all windows | M (ii) M3's way")
        for v in VARS[1:]:
            lm, ls = ratio(D, (D[f"L_{v}"] - D.L_base).to_numpy(), np.ones(len(D))); fm, fs = ratio(D, (D[f"F_{v}"] - D.F_base).to_numpy(), np.ones(len(D))); mm_, ms = ratio(D, (D[f"M_{v}"] - D.M_base).to_numpy(), D.nsel.to_numpy(float))
            print(f"  {v:7s} L {lm:+.4f} ({ls:.4f}) | F {fm:+.4f} ({fs:.4f}) | M {mm_:+.4f} ({ms:.4f})", flush=True)
    print("--- March dev, judge, variant minus base (flow A)")
    for v in VARS[1:]:
        s = m3v2.summary(Dm, "base", v, "private"); print(f"  {v:7s} raw {(Dm[v] - Dm.base).mean():+.4f} | judge {s['t2_weighted']:+.4f} (SE {s['se']:.4f}, z {s['t2_weighted'] / max(s['se'], 1e-9):+.2f}, /1.25 {s['t2_weighted'] / max(s['se'], 1e-9) / 1.25:+.2f}) | March composition {m3v2.summary(Dm, 'base', v, 'validation', n_boot=500, n_draw=200)['t2_weighted']:+.4f}", flush=True)
