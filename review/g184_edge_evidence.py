"""G184 (11/01, ledger #142 stage 1): onset windows on days 1-3 of January and February, the final A flow (G170 path). base = the rule as produced: earlier days of the same month only, k expectation. ext = three earlier days always, reaching into the month before; the train panels are continuous across months, the month before is seen through its Task 1 mask.
Also B 1.0 and B 0.5 on the same windows, to read the rule itself there. Identity: base against g119's PE. M3 dev has only 2 windows on those days, so March is read on the LB only.
usage: TFB_M3VER=v3 py -B -X utf8 g184_edge_evidence.py"""
import sys, pickle
R = r"C:/Users/user/tfb/review/"; sys.dont_write_bytecode = True
src = open(R + "g119_month_postproc.py", encoding="utf-8").read().split('if __name__ == "__main__":')[0]
sys.argv = [sys.argv[0], "run"]; exec(compile(src, "g119", "exec"), globals())
OOF = [pickle.load(open(W + f"oof_q25_nomem_s{s}.pkl", "rb")) for s in (3, 4)]; FF = phys_feats(); rows = []


def pred(c, pa, t, k, B):
    s, nd = seen(c, t, k, -6, 12); pb = pa.copy()
    if nd:
        pb[-1] = pb[-1] * np.where(G.dil(s, 2), 1.0, B)
    return decode(pb)[-1]


def i1(s_, y, e):
    u = ((s_ | y) & e).sum(); return 1.0 if u == 0 else float((s_ & y & e).sum() / u)


for mo in MONTHS:
    mdl = phys_lomo(FF, mo); F_m = rarity_lomo(mo); ks = np.flatnonzero(T.META["is_sel"] & (MON == mo) & (T.META["cond"] == "queue_onset"))
    for p in sorted({str(T.META["panel"][k]) for k in ks}):
        P = load(p, "train"); key = (mo, p, "mask"); CTX.clear(); sim_blank.cache_clear(); CTX[key] = Ctx(key, P, np.where(P.blank, np.nan, P.V), P.V); c = CTX[key]; ev = month_events(c, p, P, mo, F_m)
        for k in ks[np.array([str(T.META["panel"][k]) == p for k in ks])]:
            t = int(T.META["t"][k]); dom = int(P.times[t][8:10]) - 1
            if dom > 2:
                continue
            lab, el = T.LAB[k]; Xw = np.asarray(XN[OFFN[k]:OFFN[k + 1]]); y, e = lab[5].astype(bool), el[5].astype(bool); p_ = (OOF[0][k] + OOF[1][k]) / 2; a = PO.adaptive_a(ev, dom); pa = p_.copy(); pa[5] = (1 - a) * p_[5] + a * mdl.predict(PO.feats(Xw)); r = dict(w=int(k), month=mo, panel=p, day=dom, base=0.0, b10=i1(decode(pa)[-1], y, e), ext=i1(pred(c, pa, t, 3, 0.25), y, e))
            for kk, pr in m3v2.kdist(dom, cap=3).items():
                r["base"] += pr * i1(pred(c, pa, t, kk, 0.25), y, e)
            rows.append(r)
        load.cache_clear()
D = pd.DataFrame(rows); D.to_csv(W + "g184_edge.csv", index=False); ref = pd.read_csv(W + "g119_month_postproc.csv", dtype={"w": str}); ref = ref[ref.vis == "mask"].set_index(["month", "w"]).PE; print("identity", float(max(abs(ref.loc[(r.month, str(r.w))] - r.base) for r in D.itertuples())), len(D), D.groupby("day").size().to_dict(), flush=True)
for nm, d in list(D.groupby("month")) + [("both", D)]:
    for c_ in ("ext", "b10"):
        x = (d[c_] - d.base).to_numpy(); ids = pd.factorize(d.panel + "_" + d.day.astype(str))[0]; M = np.bincount(ids, x); N = np.bincount(ids); Rm = np.random.default_rng(0).multinomial(len(M), np.full(len(M), 1 / len(M)), size=2000); print(nm, c_, len(d), f"{x.mean():+.4f} (SE {((Rm @ M) / (Rm @ N)).std():.4f}) changed {(x != 0).sum()}")
