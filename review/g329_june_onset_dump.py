"""G329 (11/05, ledger #262 data): June 2030 onset windows, G321's flow (cvf fold 2030-06), the T+30 row just before best_set per k with the
labels -> work/t2/g329_jun_onset_rows.pkl. From G321: the signs of the evidence penalty and the ongoing temperature on a month the networks did not
see, read with official-type inputs. June 2030 selector windows (330), the networks cvf_attnglob s0 / s1 fold 2030-06 (trained without
2030-06 .. 08), inputs as the official windows (G320: history as released, MASKHIST=0), the "mask" context of G119 / G170 (the Task 1 blank
hidden outside the window's own rows; the earlier days' Task 2 blanks simulated as G119 does). Onset (G170's train_months flow, the
leave-one-month-out physics model, L labels): penalty 1.0 - 0.25. Ongoing (G119's flow as G287): temperature 1.0 and 1.2 - 0.8.
Cluster bootstrap (panel x day). usage: TFB_M3VER=v3 py -B -X utf8 g321_june_signs.py"""
import sys, pickle
s = open(r"C:/Users/user/tfb/review/g170_onset_instrument.py", encoding="utf-8").read()


def sub(a, b, n=1):
    global s
    assert s.count(a) == n, (a[:60], s.count(a))
    s = s.replace(a, b)


sub('p_ = (OOF[0][k] + OOF[1][k]) / 2;', 'p_ = (CV_[0][str(k)] + CV_[1][str(k)]) / 2;')
sub('    return {v: decode(x)[-1] for v, x in pm.items()}', '    out_ = {v: decode(x)[-1] for v, x in pm.items()}; out_["_b"] = b; return out_')
sub('                    S = sets(c, p_, ph, a, t, kk)\n', '                    S = sets(c, p_, ph, a, t, kk); DJ_.setdefault(int(k), dict(y=yL, e=eL, panel=p, t=t, month=mo, C=[]))["C"].append((pr, S["_b"][-1]))\n')
sub('    for mo in MONTHS:\n        mdl = phys_lomo', '    for mo in ("2030-06",):\n        mdl = phys_lomo')
sub('def march(limit=None):', 'DJ_ = {}; CV_ = [pickle.load(open(W + f"probs_junmh0__cvf_attnglob_s{s_}_fold2030-06.pkl", "rb")) for s_ in (0, 1)]\n\n\ndef march(limit=None):')
sub("t0 = TM(); Dm = march(lim)", "t0 = TM(); Dm = None")
sub("t0 = TM(); Dt = train_months(lim)", '''t0 = TM(); Dt = train_months(lim); pickle.dump(DJ_, open(W + "g329_jun_onset_rows.pkl", "wb")); print("June onset rows dumped", len(DJ_), "| identity against g321:", float(np.abs(Dt.set_index("w").L_base - pd.read_csv(W + "g321_jun_onset.csv").set_index("w").L_base.reindex(Dt.w).to_numpy()).max()), flush=True); sys.exit()
    cl = pd.factorize(Dt.panel + "_" + Dt.day.astype(str))[0]; Rm = np.random.default_rng(0).multinomial(cl.max() + 1, np.full(cl.max() + 1, 1 / (cl.max() + 1)), size=2000)
    bse = lambda d_: float(((Rm @ np.bincount(cl, d_)) / (Rm @ np.bincount(cl))).std())
    d_ = (Dt["L_B1.0"] - Dt.L_base).to_numpy(); print(f"June onset {len(Dt)} windows, base {Dt.L_base.mean():.4f}: penalty 1.0 - 0.25 {d_.mean():+.5f} (SE {bse(d_):.5f}, z {d_.mean() / max(bse(d_), 1e-12):+.2f}, changed {int((np.abs(d_) > 1e-12).sum())})", flush=True)
    import importlib.util as iu_
    from t2_model import best_set as bs_
    KS_ = [k for k in np.flatnonzero(T.META["is_sel"] & (MON == "2030-06")) if str(T.META["cond"][k]) == "queue_ongoing"]; rows_ = []
    tmp_ = lambda p, T_: sg(lg(p) / T_)
    for p in sorted({str(T.META["panel"][k]) for k in KS_}):
        P = load(p, "train"); key = ("2030-06", p, "mask"); CTX.clear(); sim_blank.cache_clear(); CTX[key] = Ctx(key, P, np.where(P.blank, np.nan, P.V), P.V); c = CTX[key]
        for k in [k for k in KS_ if str(T.META["panel"][k]) == p]:
            t = int(T.META["t"][k]); dom = int(P.times[t][8:10]) - 1; lab, el = T.LAB[k]; y, e = lab.astype(bool), el.astype(bool)
            lq = np.asarray(XN[OFFN[k]:OFFN[k + 1]][:, G.CI], float) == 1; p_ = (CV_[0][str(k)] + CV_[1][str(k)]) / 2; cone = cone_parts(c, t, lq, P.length); r = dict(w=int(k), panel=p, day=dom)
            for T_ in (0.8, 1.0, 1.2):
                r[f"t{T_}"] = 0.0
            for kk, pr in m3v2.kdist(dom, cap=3).items():
                pc = p_ * cone_f(c, cone, t, kk); s_, nd = seen(c, t, kk, -18, 12); pb = pc * np.where(G.dil(s_, 1), 1.0, 0.85)[None, :] if nd else pc.copy()
                for T_ in (0.8, 1.0, 1.2):
                    r[f"t{T_}"] += pr * iou(far_gate(c, decode(tmp_(pb, T_)), t, kk, lq), y, e, "queue_ongoing")
            rows_.append(r)
        load.cache_clear()
    Do = pd.DataFrame(rows_); Do.to_csv(W + "g321_jun_ongoing.csv", index=False); cl = pd.factorize(Do.panel + "_" + Do.day.astype(str))[0]; Rm = np.random.default_rng(0).multinomial(cl.max() + 1, np.full(cl.max() + 1, 1 / (cl.max() + 1)), size=2000)
    for T_ in (1.0, 1.2):
        d_ = (Do[f"t{T_}"] - Do["t0.8"]).to_numpy(); print(f"June ongoing {len(Do)} windows, T 0.8 {Do['t0.8'].mean():.4f}: T {T_} - 0.8 {d_.mean():+.5f} (SE {bse(d_):.5f}, z {d_.mean() / max(bse(d_), 1e-12):+.2f})", flush=True)
    sys.exit()
    Dt = None''')
sub('    print(f"M3 dev {len(Dm)} windows {TM() - t0:.0f}s | identity against g108 A: {np.abs(o - Dm.base.to_numpy()).max():.2e}", flush=True); ', '    ')
sub("o = pd.read_csv(W + \"g108_rectangles.csv\").set_index(\"w\").A.reindex(Dm.w).to_numpy()", "o = None")
exec(compile(s, "g170p19", "exec"))
