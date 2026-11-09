"""G155 (11/01, ledger #115, strategy T2_DEEP_REVIEW 1-3): the onset evidence penalty switched by a precursor flag. Links with no same-month
evidence get T+30 x B; B is 0.25 in the finals. Here B = B_low on windows flagged "no precursor", B_high on the others, on six cells
B_low in {0.1, 0.25} x B_high in {0.5, 0.7, 1.0}; only the decode changes, the saved probabilities do not.
The flag of J11 (G117) reads the label set Y at T+30, so it cannot enter a forecast (749601). Two flags, the J11 tests (a) speed, (b) upstream flow, (c) rare site:
  flag L (the candidate, <= T): read on S0 = the set decoded with no evidence rule (G120's reading of the tests on a chosen set), empty S0 -> not flagged
  flag O (oracle, read only for "is there a pair non-negative on all three months"): read on Y as G117, empty Y -> not flagged
vff of test (a) = the link's free-flow level on the earlier days of the same month (G154: it moves by sd 1-1.5 % in the redrawn months), the training value where a link has < 50 free cells.
L0 / O0 = the same flags with the training vff (G120, G117), counted only.
Flows: A = the final A onset flow (NN + physics onset with adaptive a), B = the final B onset flow (NN only).
  M3 dev : 100 onset windows, q24f 8 seeds CF5 (G120), identity A to g108 column A (hold-out days are not read)
  Jan, Feb: q25_nomem seeds 3, 4 OOF, visibility "mask", identity A / B to g119's PE / E. vff, profile, rarity and physics leave the month out.
usage: TFB_M3VER=v3 py -B -X utf8 g155_onset_precursor_b.py [time | run]"""
import os, sys, pickle, warnings
R = r"C:/Users/user/tfb/review/"; sys.dont_write_bytecode = True; RUN = (sys.argv[1:] or ["run"])[0]
src = open(R + "g119_month_postproc.py", encoding="utf-8").read().split('if __name__ == "__main__":')[0]
sys.argv = [sys.argv[0], "run"]; exec(compile(src, "g119", "exec"), globals())
import importlib.util
sp = importlib.util.spec_from_file_location("g120", R + "g120_onset_gate.py"); g2 = importlib.util.module_from_spec(sp); sp.loader.exec_module(g2)
from t2_demand import profile

BS = (0.1, 0.25, 0.5, 0.7, 1.0); LOW = (0.1, 0.25); HIGH = (0.5, 0.7, 1.0); CELLS = [(l, h) for l in LOW for h in HIGH]; K = 22
name = lambda f, l, h: f"{f}_l{l}_h{h}"


def sets_by_b(c, pp, t, k):
    """the decoded (6, n) sets for every B; with no earlier day the rule does nothing (onset_pred)"""
    s, nd = seen(c, t, k, -6, 12); ev = G.dil(s, 2); out = {}
    for b in BS:
        pb = pp.copy()
        if nd:
            pb[-1] = pb[-1] * np.where(ev, 1.0, b)
        out[b] = decode(pb)
    return out


def no_precursor(S, m, P, stats):
    return all(g2.precursor(S, m, P, *stats)) if S.any() else False


ASOF = {}


def vff_asof(c, P, r0, dom, vff0):
    k = (c.key, dom)
    if k not in ASOF:
        rr = slice(r0, r0 + dom * 288); V_ = c.ns.V[rr]; ff = np.where(P.E[rr] & (V_ >= 0.85 * P.vf[None, :]), V_, np.nan); n = np.isfinite(ff).sum(axis=0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore"); v = np.nanmedian(ff, axis=0) if len(ff) else np.full(P.n, np.nan)
        ASOF[k] = np.where(n >= 50, v, vff0)
    return ASOF[k]


def window(c, P, stats, p_, ph, a, t, panel, y, e, dom, r0):
    pa = p_.copy(); pa[5] = (1 - a) * p_[5] + a * ph; m = dict(t=t, panel=panel); r = {}
    st_a = ({panel: vff_asof(c, P, r0, dom, stats[0][panel])}, *stats[1:])
    Y = y[-1] & e[-1]; O = no_precursor(Y, m, P, st_a); r["flag_O"] = O; r["flag_O0"] = no_precursor(Y, m, P, stats)
    for fl, pp in (("A", pa), ("B", p_)):
        S0 = decode(pp)[-1]; L = no_precursor(S0, m, P, st_a); r[f"flag_L_{fl}"] = L; r[f"flag_L0_{fl}"] = no_precursor(S0, m, P, stats); sc = {b: 0.0 for b in BS}
        for k, pr in m3v2.kdist(dom, cap=3).items():
            sets = sets_by_b(c, pp, t, k)
            for b in BS:
                sc[b] += pr * iou(sets[b], y, e, "queue_onset")
        r[f"{fl}_u"] = sc[0.25]
        for l, h in CELLS:
            r[name(fl + "L", l, h)] = sc[l] if L else sc[h]; r[name(fl + "O", l, h)] = sc[l] if O else sc[h]
    return r


def march(limit=None):
    PN = g2.nn_blend(); phys = lgb.Booster(model_file=W + "physon_base.txt"); F_m = PO.rarity_table(); ev = PO.month_events("validation", F_m)
    VFF, PROF, UP = g2.panel_stats(); stats = (VFF, PROF, UP, F_m); rows = []
    for w in g2.DEV[:limit]:
        m = G.META[w]; key = ("val", m["panel"]); i = G.POS[w]; Xw = np.asarray(G.X[G.OFF[i]:G.OFF[i + 1]]); P = load(m["panel"], "validation")
        if key not in CTX:
            CTX[key] = Ctx(key, P, P.V)
        r = window(CTX[key], P, stats, PN[w], phys.predict(PO.feats(Xw)), PO.adaptive_a(ev, m["day"]), m["t"], m["panel"], m["y"].T.astype(bool), m["m"].T.astype(bool), m["day"], 0)
        rows.append(dict(w=w, month="2031-03", cond=m["cond"], panel=m["panel"], day=m["day"], hour=(m["t"] % 288) // 12, **r))
    return pd.DataFrame(rows)


def up_dir(P):
    li = {l: i for i, l in enumerate(P.links)}; sg = []
    for j, s in enumerate(P.topo.incoming_link_ids.fillna("").astype(str)):
        sg += [np.sign(li[x] - j) for x in s.split(";") if x in li]
    return int(np.sign(np.median(sg)))


def train_months(limit=None):
    OOF = [pickle.load(open(W + f"oof_q25_nomem_s{s}.pkl", "rb")) for s in (3, 4)]; FF = phys_feats() if limit is None else None; rows = []
    for mo in MONTHS:
        mdl = phys_lomo(FF, mo) if FF is not None else lgb.Booster(model_file=W + "physon_base.txt"); F_m = rarity_lomo(mo)
        ks = np.flatnonzero(T.META["is_sel"] & (MON == mo) & (T.META["cond"] == "queue_onset"))
        for p in sorted({str(T.META["panel"][k]) for k in ks}):
            P = load(p, "train"); key = (mo, p, "mask"); CTX.clear(); sim_blank.cache_clear()
            CTX[key] = Ctx(key, P, np.where(P.blank, np.nan, P.V), P.V); c = CTX[key]; ev = month_events(c, p, P, mo, F_m)
            inmo = np.char.startswith(P.times.astype(str), mo); r0 = int(np.flatnonzero(inmo)[0]); other = ~inmo[:, None]
            ff = np.where(P.E & other & (P.V >= 0.85 * P.vf[None, :]), P.V, np.nan)
            stats = ({p: np.nanmedian(ff, axis=0)}, {p: profile(P, exclude_month=mo)}, {p: up_dir(P)}, F_m)
            for k in ks[np.array([str(T.META["panel"][k]) == p for k in ks])]:
                t = int(T.META["t"][k]); dom = int(P.times[t][8:10]) - 1; lab, el = T.LAB[k]; Xw = np.asarray(XN[OFFN[k]:OFFN[k + 1]])
                r = window(c, P, stats, (OOF[0][k] + OOF[1][k]) / 2, mdl.predict(PO.feats(Xw)), PO.adaptive_a(ev, dom), t, p, lab.astype(bool), el.astype(bool), dom, r0)
                rows.append(dict(w=int(k), month=mo, cond="queue_onset", panel=p, day=dom, hour=(t % 288) // 12, **r))
                if limit and len(rows) >= limit:
                    return pd.DataFrame(rows)
            load.cache_clear()
    return pd.DataFrame(rows)


def clse(d, cl, n_boot=2000, seed=0):
    ids = pd.factorize(cl)[0]; M = np.bincount(ids, d); N = np.bincount(ids)
    Rm = np.random.default_rng(seed).multinomial(len(M), np.full(len(M), 1 / len(M)), size=n_boot); return float(((Rm @ M) / (Rm @ N)).std())


if __name__ == "__main__":
    t0 = TM(); lim = 6 if RUN == "time" else None; Dm = march(lim); tm = TM() - t0
    o = pd.read_csv(W + "g108_rectangles.csv").set_index("w").A.reindex(Dm.w).to_numpy()
    print(f"M3 dev {len(Dm)} windows in {tm:.0f}s | identity against g108 A: max |d| {np.abs(o - Dm.A_u.to_numpy()).max():.2e} | mean A_u {Dm.A_u.mean():.4f}", flush=True)
    t0 = TM(); Dt = train_months(lim); tt = TM() - t0; ref = pd.read_csv(W + "g119_month_postproc.csv", dtype={"w": str}); ref = ref[ref.vis == "mask"]
    ref = ref.assign(w=ref.w.astype(int) if RUN == "x" else ref.w).set_index(["month", "w"])
    ix = pd.MultiIndex.from_arrays([Dt.month, Dt.w.astype(str)]); d = [np.abs(ref.PE.reindex(ix).to_numpy() - Dt.A_u.to_numpy()), np.abs(ref.E.reindex(ix).to_numpy() - Dt.B_u.to_numpy())]
    print(f"Jan, Feb {len(Dt)} windows in {tt:.0f}s | identity against g119 PE / E: max |d| {np.nanmax(d[0]):.2e} / {np.nanmax(d[1]):.2e}, missing {int(np.isnan(d[0]).sum())}", flush=True)
    if RUN == "time":
        sys.exit()
    if np.abs(o - Dm.A_u.to_numpy()).max() > 1e-12 or max(np.nanmax(d[0]), np.nanmax(d[1])) > 1e-12 or np.isnan(d[0]).any():
        print("IDENTITY FAILED: stop"); sys.exit(1)
    Dm.to_csv(W + "g155_onset_b_m3.csv", index=False); Dt.to_csv(W + "g155_onset_b_train.csv", index=False)
    emax = float(np.sqrt(2 * np.log(K)) - 0.5); A = pd.concat([Dm, Dt], ignore_index=True)
    for mo, g_ in A.groupby("month"):
        print(f"{mo}: windows {len(g_)} | flagged L(A) {int(g_.flag_L_A.sum())} L(B) {int(g_.flag_L_B.sum())} O {int(g_.flag_O.sum())}, training vff L0(A) {int(g_.flag_L0_A.sum())} O0 {int(g_.flag_O0.sum())} | both L(A) and O {int((g_.flag_L_A & g_.flag_O).sum())} | IoU A {g_.A_u.mean():.4f} B {g_.B_u.mean():.4f}")
    for fl in ("AL", "BL", "AO", "BO"):
        print(f"--- flow {fl[0]}, flag {'L (<= T)' if fl[1] == 'L' else 'O (oracle, not adoptable)'}: difference from the uniform 0.25")
        for l, h in CELLS:
            c_ = name(fl, l, h); u = fl[0] + "_u"; out = []; signs = []
            for mo in ("2031-03",) + MONTHS:
                g_ = A[A.month == mo]; dd = (g_[c_] - g_[u]).to_numpy(); se = clse(dd, g_.panel + "_" + g_.day.astype(str)); out.append(f"{mo[-2:]} {dd.mean():+.4f} ({se:.4f})"); signs.append(dd.mean() >= 0)
            s = m3v2.summary(Dm, u, c_, "private"); z = s["t2_weighted"] / s["se"]
            print(f"  B_low {l} B_high {h}: " + " | ".join(out) + f" | judge {s['t2_weighted']:+.4f} SE {s['se']:.4f} z {z:+.2f} /1.25 {z / 1.25:+.2f} k{K} {z / 1.25 - emax:+.2f} | all three >= 0: {all(signs)}", flush=True)
