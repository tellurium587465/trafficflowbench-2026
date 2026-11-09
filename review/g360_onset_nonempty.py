"""G360 (11/05, ledger #287): onset decode conditioned on a queue at T+30 (src/t2_onsetcond.condition, the production function). G170's flow:
M3 v3 dev onset 100 windows with A17's onset (0.8 q24f + 0.2 pcL 4 seeds); January / February 317 onset windows with the OOF flow; mode
jun80: the 80 official June windows via G342b's augmented evaluator (cvf fold 2030-06). Variant "ne": after the penalty (b = rule(pa, 0.25)),
b's T+30 row replaced by its marginals given >= 1 queued cell (seed crc32 of the window key), then the decode as the base. Line (#287): the
three readings stacked positive, none worse than -2 SE, April rows change. usage: TFB_M3VER=v3 py -B -X utf8 g360_onset_nonempty.py <m3jf | jun80>"""
import os, sys, pickle
MODE = sys.argv[1]; R0 = "C:/Users/user/tfb/"; W = R0 + "work/t2/"
if MODE == "jun80":
    g = open(R0 + "review/g342b_jun80_run.py", encoding="utf-8").read(); g = g[:g.index("src = open(")]; sys.argv = ["g342b", "none"]; exec(compile(g, "g342b_head", "exec"))
sys.argv = sys.argv[:1]; sys.path.insert(0, r"C:/Users/user/tfb_a8/src")
import t2_onsetcond as OC
s = open(R0 + "review/g170_onset_instrument.py", encoding="utf-8").read()


def sub(a, b, n=1):
    global s
    assert s.count(a) == n, (a[:70], s.count(a))
    s = s.replace(a, b)


sub('    return {v: decode(x)[-1] for v, x in pm.items()}', '    pm["ne"] = OC.condition(b, CW_[0])[0]\n    return {v: decode(x)[-1] for v, x in pm.items()}')
sub('"hid", "hidf", "ac"); ND = 8', '"hid", "hidf", "ac", "ne"); ND = 8')
sub('            S = sets(CTX[key], PN[w], ph, a, m["t"], k, PO.adaptive_a(ev2, m["day"]))\n',
    '            CW_[0] = w; S = sets(CTX[key], 0.8 * PN[w] + 0.2 * sum(PC_[j][w] for j in range(4)) / 4, ph, a, m["t"], k, PO.adaptive_a(ev2, m["day"]))\n')
sub('                    S = sets(c, p_, ph, a, t, kk)\n', '                    CW_[0] = int(k); S = sets(c, p_, ph, a, t, kk)\n')
sub('def march(limit=None):', 'PC_ = [pickle.load(open(W + f"probs_m3v3cap__pcL_s{s_}_fold2030-12.pkl", "rb")) for s_ in range(4)]; CW_ = [None]\n\n\ndef march(limit=None):')
if MODE == "jun80":
    sub('p_ = (OOF[0][k] + OOF[1][k]) / 2;', 'p_ = (CV_[0][str(k)] + CV_[1][str(k)]) / 2;')
    sub('    for mo in MONTHS:\n        mdl = phys_lomo', '    for mo in ("2030-06",):\n        mdl = phys_lomo')
    sub('def march(limit=None):', 'CV_ = [pickle.load(open(W + f"probs_jun80__cvf_attnglob_s{s_}_fold2030-06.pkl", "rb")) for s_ in (0, 1)]\n\n\ndef march(limit=None):')
    sub("t0 = TM(); Dm = march(lim)", "t0 = TM(); Dm = None")
    sub('    print(f"M3 dev {len(Dm)} windows {TM() - t0:.0f}s | identity against g108 A: {np.abs(o - Dm.base.to_numpy()).max():.2e}", flush=True); ', '    ')
    sub("o = pd.read_csv(W + \"g108_rectangles.csv\").set_index(\"w\").A.reindex(Dm.w).to_numpy()", "o = None")
else:
    sub("o = pd.read_csv(W + \"g108_rectangles.csv\").set_index(\"w\").A.reindex(Dm.w).to_numpy()", "o = Dm.base.to_numpy()")
sub("t0 = TM(); Dt = train_months(lim)", f'''t0 = TM(); Dt = train_months(lim); Dt.to_csv(W + "g360_{MODE}_t.csv", index=False)
    def cse(x, g):
        gi = pd.factorize(g)[0]; G = gi.max() + 1; Rm = np.random.default_rng(0).multinomial(G, np.full(G, 1 / G), size=4000); return float(((Rm @ np.bincount(gi, x)) / (Rm @ np.bincount(gi))).std())
    x = (Dt.L_ne - Dt.L_base).to_numpy(); lab_ = "June official onset" if Dm is None else "Jan / Feb onset"
    print(f"{{lab_}}: base {{Dt.L_base.mean():.4f}}, ne - base {{x.mean():+.5f}} (route SE {{cse(x, Dt.panel.astype(str)):.5f}}, panel x day SE {{cse(x, Dt.panel.astype(str) + '_' + Dt.day.astype(str)):.5f}}, changed {{int((np.abs(x) > 1e-12).sum())}}, up {{int((x > 1e-12).sum())}}, down {{int((x < -1e-12).sum())}})", flush=True)
    if Dm is not None:
        Dm = Dm.rename(columns={{"ne": "nonemp"}}); Dm.to_csv(W + "g360_m3.csv", index=False); sm = m3v2.summary(Dm, "base", "nonemp", "private")
        print(f"M3 onset: base {{Dm.base.mean():.4f}}, ne raw {{(Dm.nonemp - Dm.base).mean():+.5f}}, judge {{sm['t2_weighted']:+.5f}} (SE {{sm['se']:.5f}}, z {{sm['t2_weighted'] / max(sm['se'], 1e-12):+.2f}}, changed {{int((Dm.nonemp != Dm.base).sum())}})", flush=True)
    sys.exit()
    Dt = None''')
exec(compile(s, "g170_g360", "exec"))
