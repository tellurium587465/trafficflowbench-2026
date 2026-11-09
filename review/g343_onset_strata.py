"""G343 (11/05, ledger #274, measurement only): onset strata. G170's flow as G339 / G335 (M3 v3 dev onset 100 windows with A17's onset
= 0.8 q24f + 0.2 pcL 4 seeds; January / February 317 with the OOF flow; mode jun: the June 2030 replica selector onset windows 167 with
cvf fold 2030-06 as G321). Per window: IoU of the base (penalty 0.25) and of B1.0 (no penalty), and (iv) "evidence acts" = the k-weighted
share of draws where the decoded T+30 set with the penalty differs from the one without it. Strata (all known at T): (i) the official-type
order = dense rank of the window's day among the days with a window of that panel x month (1-5, 6+; the official windows are one a day,
ranks 1-5 = days 1-5 mostly), (ii) day 1-2 / 3+, (iii) T's hour: morning <= 8 / midday 9-12 / afternoon >= 13 (the official onsets lie
at hours 4-16), (iv) evidence acts (> 0.5) / not. Per stratum: n, IoU, B1.0 - base. April 40: counts by (i)-(iii).
usage: TFB_M3VER=v3 py -B -X utf8 g343_onset_strata.py [m3jf | jun]"""
import sys, pickle
MODE = (sys.argv[1:] or ["m3jf"])[0]; sys.argv = sys.argv[:1]
s = open(r"C:/Users/user/tfb/review/g170_onset_instrument.py", encoding="utf-8").read()


def sub(a, b, n=1):
    global s
    assert s.count(a) == n, (a[:60], s.count(a))
    s = s.replace(a, b)


sub('    return {v: decode(x)[-1] for v, x in pm.items()}', '    out_ = {v: decode(x)[-1] for v, x in pm.items()}; out_["_chg"] = float((out_["base"] != out_["B1.0"]).any()); return out_')
sub('            for v in VARS:\n                sc[v] += pr * iou1(S[v], y, e)\n', '            CH_[w] = CH_.get(w, 0.0) + pr * S["_chg"]\n            for v in VARS:\n                sc[v] += pr * iou1(S[v], y, e)\n')
sub('                    S = sets(c, p_, ph, a, t, kk)\n', '                    S = sets(c, p_, ph, a, t, kk); CH_[int(k)] = CH_.get(int(k), 0.0) + pr * S["_chg"]\n')
sub('def march(limit=None):', 'PC_ = [pickle.load(open(W + f"probs_m3v3cap__pcL_s{s_}_fold2030-12.pkl", "rb")) for s_ in range(4)]; CH_ = {}\n\n\ndef march(limit=None):')
sub('            S = sets(CTX[key], PN[w], ph, a, m["t"], k, PO.adaptive_a(ev2, m["day"]))\n',
    '            S = sets(CTX[key], 0.8 * PN[w] + 0.2 * sum(PC_[j][w] for j in range(4)) / 4, ph, a, m["t"], k, PO.adaptive_a(ev2, m["day"]))\n')
if MODE == "jun":
    sub('p_ = (OOF[0][k] + OOF[1][k]) / 2;', 'p_ = (CV_[0][str(k)] + CV_[1][str(k)]) / 2;')
    sub('    for mo in MONTHS:\n        mdl = phys_lomo', '    for mo in ("2030-06",):\n        mdl = phys_lomo')
    sub('def march(limit=None):', 'CV_ = [pickle.load(open(W + f"probs_junmh0__cvf_attnglob_s{s_}_fold2030-06.pkl", "rb")) for s_ in (0, 1)]\n\n\ndef march(limit=None):')
    sub("t0 = TM(); Dm = march(lim)", "t0 = TM(); Dm = None")
    sub('    print(f"M3 dev {len(Dm)} windows {TM() - t0:.0f}s | identity against g108 A: {np.abs(o - Dm.base.to_numpy()).max():.2e}", flush=True); ', '    ')
    sub("o = pd.read_csv(W + \"g108_rectangles.csv\").set_index(\"w\").A.reindex(Dm.w).to_numpy()", "o = None")
else:
    sub("t0 = TM(); Dm = march(lim)", "t0 = TM(); Dm = march(lim); P8 = pd.read_csv(W + 'g280_m3.csv').set_index('w'); print('identity base (A17 onset) against g280 pc4:', float(np.abs(Dm.set_index('w').base - P8.pc4.reindex(Dm.w).to_numpy()).max()), flush=True)")
    sub("o = pd.read_csv(W + \"g108_rectangles.csv\").set_index(\"w\").A.reindex(Dm.w).to_numpy()", "o = Dm.base.to_numpy()")
sub("t0 = TM(); Dt = train_months(lim)", '''t0 = TM(); Dt = train_months(lim)
    def strata(D_, b_, d_, lab):
        D_ = D_.copy(); D_["chg"] = D_.w.map(lambda x: CH_.get(x if isinstance(x, str) else int(x), np.nan)); D_["iou"] = D_[b_]; D_["dB"] = D_[d_] - D_[b_]
        D_["rank"] = D_.groupby(D_.grp).day.rank(method="dense").astype(int).clip(upper=6).map(lambda r: str(r) if r <= 5 else "6+")
        D_["band"] = np.where(D_.day <= 1, "d1-2", "d3+"); D_["tod"] = np.where(D_.hr <= 8, "morning", np.where(D_.hr <= 12, "midday", "afternoon")); D_["ev"] = np.where(D_.chg > 0.5, "acts", "no")
        D_.to_csv(W + f"g343_{lab}.csv", index=False); print(f"=== {lab}: {len(D_)} windows, IoU {D_.iou.mean():.4f}, B1.0 - base {D_.dB.mean():+.4f}", flush=True)
        for a_ in ("rank", "band", "tod", "ev"):
            g_ = D_.groupby(a_).agg(n=("iou", "size"), iou=("iou", "mean"), dB=("dB", "mean"))
            print(f"  {a_}: " + " | ".join(f"{i} n{int(r.n)} IoU {r.iou:.3f} dB {r.dB:+.4f}" for i, r in g_.iterrows()), flush=True)
    if Dm is not None:
        Dm["grp"] = Dm.panel; Dm["hr"] = [(int(G.META[w]["t"]) % 288) // 12 for w in Dm.w]; strata(Dm, "base", "B1.0", "M3")
    Dt["grp"] = Dt.panel + "_" + Dt.month; Dt["hr"] = [(int(T.META["t"][int(k)]) % 288) // 12 for k in Dt.w]; strata(Dt, "L_base", "L_B1.0", "JUN" if Dm is None else "JF")
    if Dm is not None:
        from cachelib import T2_PANELS as TP_
        from common import REL as REL_
        A_ = pd.concat([pd.read_csv(REL_ / "task2" / p_ / "private" / "window_index.csv") for p_ in TP_]); A_ = A_[A_.condition == "queue_onset"].copy(); fo = pd.to_datetime(A_.forecast_origin)
        A_["rank"] = A_.groupby("panel").forecast_origin.rank(method="dense").astype(int).astype(str); A_["band"] = np.where(fo.dt.day <= 2, "d1-2", "d3+"); h_ = fo.dt.hour; A_["tod"] = np.where(h_ <= 8, "morning", np.where(h_ <= 12, "midday", "afternoon"))
        print("April 40 onset counts: " + " | ".join(f"{a_}: " + ", ".join(f"{i} {n}" for i, n in A_[a_].value_counts().sort_index().items()) for a_ in ("rank", "band", "tod")), flush=True)
    sys.exit()
    Dt = None''')
exec(compile(s, "g170p343", "exec"))
