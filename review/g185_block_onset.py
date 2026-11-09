"""G185 (11/01, ledger #144, strategy T2_DEEP_REVIEW 18, rereading G46): the generator's blocks: adjacent links split the queue at rate <= 0.05 over pairs congested >= 30 rows, from the training months (January, February: without the month). Onset T+30 probabilities pooled in the block, before best_set, after the evidence rule: b100 = block mean, b50 = half way. Final A's onset flow, M3 dev 100 windows and January, February. Identity: base against g108, g119.
usage: TFB_M3VER=v3 py -B -X utf8 g185_block_onset.py"""
import sys, pickle, importlib.util
R = r"C:/Users/user/tfb/review/"; sys.dont_write_bytecode = True
src = open(R + "g119_month_postproc.py", encoding="utf-8").read().split('if __name__ == "__main__":')[0]
sys.argv = [sys.argv[0], "run"]; exec(compile(src, "g119", "exec"), globals())
md = lambda nm, pa: (lambda sp: (lambda m: (sp.loader.exec_module(m), m)[1])(importlib.util.module_from_spec(sp)))(importlib.util.spec_from_file_location(nm, pa))
g2 = md("g120", R + "g120_onset_gate.py"); OOF = [pickle.load(open(W + f"oof_q25_nomem_s{s}.pkl", "rb")) for s in (3, 4)]; BK = {}


def blocks(p, excl):
    if (p, excl) not in BK:
        P = load(p, "train"); m = ~np.char.startswith(P.times.astype(str), excl) if excl else np.ones(P.T, bool); V = P.V[m]; ok = np.isfinite(V); c = (V <= P.vcut[None, :]) & ok; a, b = c[:, :-1], c[:, 1:]; both = ok[:, :-1] & ok[:, 1:]; na = ((a | b) & both).sum(0); sp = ((a ^ b) & both).sum(0) / np.maximum(na, 1); tight = (na >= 30) & (sp <= 0.05); bid = np.zeros(P.n, int)
        for i in range(1, P.n):
            bid[i] = bid[i - 1] if tight[i - 1] else bid[i - 1] + 1
        BK[(p, excl)] = bid
    return BK[(p, excl)]


def i1(s_, y, e):
    u = ((s_ | y) & e).sum(); return 1.0 if u == 0 else float((s_ & y & e).sum() / u)


def arms(c, pa, t, k, bid):
    s, nd = seen(c, t, k, -6, 12); pb = pa.copy()
    if nd:
        pb[-1] = pb[-1] * np.where(G.dil(s, 2), 1.0, 0.25)
    mean = np.bincount(bid, pb[-1]) / np.bincount(bid); pm = mean[bid]; o = {"base": decode(pb)[-1]}
    for nm, a_ in (("b50", 0.5), ("b100", 1.0)):
        x = pb.copy(); x[-1] = (1 - a_) * pb[-1] + a_ * pm; o[nm] = decode(x)[-1]
    return o


PN = g2.nn_blend(); phys = lgb.Booster(model_file=W + "physon_base.txt"); ev = PO.month_events("validation", PO.rarity_table()); rows = []
for w in g2.DEV:
    m = G.META[w]; key = ("val", m["panel"]); Pv = load(m["panel"], "validation")
    if key not in CTX:
        CTX[key] = Ctx(key, Pv, Pv.V)
    i = G.POS[w]; Xw = np.asarray(G.X[G.OFF[i]:G.OFF[i + 1]]); ph = phys.predict(PO.feats(Xw)); a = PO.adaptive_a(ev, m["day"]); y, e = m["y"].T.astype(bool)[-1], m["m"].T.astype(bool)[-1]; pa = PN[w].copy(); pa[5] = (1 - a) * pa[5] + a * ph; bid = blocks(m["panel"], None); sc = dict(base=0.0, b50=0.0, b100=0.0)
    for k, pr in m3v2.kdist(m["day"], cap=3).items():
        S_ = arms(CTX[key], pa, m["t"], k, bid)
        for v in sc:
            sc[v] += pr * i1(S_[v], y, e)
    rows.append(dict(w=w, cond=m["cond"], panel=m["panel"], day=m["day"], hour=(m["t"] % 288) // 12, **sc))
Dn = pd.DataFrame(rows); o = pd.read_csv(W + "g108_rectangles.csv").set_index("w").A.reindex(Dn.w).to_numpy(); print("identity", np.abs(o - Dn.base.to_numpy()).max(), flush=True)
for v in ("b50", "b100"):
    s = m3v2.summary(Dn, "base", v, "private"); print(f"M3 {v}: April composition {s['t2_weighted']:+.4f} (SE {s['se']:.4f}, z {s['t2_weighted'] / max(s['se'], 1e-9):+.2f}) raw {(Dn[v] - Dn.base).mean():+.4f}", flush=True)
FF = phys_feats(); rows = []
for mo in MONTHS:
    mdl = phys_lomo(FF, mo); F_m = rarity_lomo(mo); ks = np.flatnonzero(T.META["is_sel"] & (MON == mo) & (T.META["cond"] == "queue_onset"))
    for p in sorted({str(T.META["panel"][k]) for k in ks}):
        bid = blocks(p, mo); P = load(p, "train"); key = (mo, p, "mask"); CTX.clear(); sim_blank.cache_clear(); CTX[key] = Ctx(key, P, np.where(P.blank, np.nan, P.V), P.V); c = CTX[key]; evm = month_events(c, p, P, mo, F_m)
        for k in ks[np.array([str(T.META["panel"][k]) == p for k in ks])]:
            t = int(T.META["t"][k]); dom = int(P.times[t][8:10]) - 1; lab, el = T.LAB[k]; Xw = np.asarray(XN[OFFN[k]:OFFN[k + 1]]); y, e = lab[5].astype(bool), el[5].astype(bool); pa = (OOF[0][k] + OOF[1][k]) / 2; a = PO.adaptive_a(evm, dom); pa = pa.copy(); pa[5] = (1 - a) * pa[5] + a * mdl.predict(PO.feats(Xw)); sc = dict(base=0.0, b50=0.0, b100=0.0)
            for kk, pr in m3v2.kdist(dom, cap=3).items():
                S_ = arms(c, pa, t, kk, bid)
                for v in sc:
                    sc[v] += pr * i1(S_[v], y, e)
            rows.append(dict(w=int(k), month=mo, panel=p, day=dom, **sc))
        load.cache_clear()
Dt = pd.DataFrame(rows)
for mo, d in Dt.groupby("month"):
    print(mo, len(d), " | ".join(f"{v} {(d[v] - d.base).mean():+.4f}" for v in ("b50", "b100")), flush=True)
