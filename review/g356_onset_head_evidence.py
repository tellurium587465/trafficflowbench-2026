"""G356 (11/05, ledger #282): finer onset evidence. G170's flow (M3 v3 dev onset 100 windows with A17's onset; January / February 317 with
the OOF flow; mode jun80: the 80 official June windows via G342b's augmented evaluator, cvf fold 2030-06). Base = the evidence rule as
is (links within +-2 of any queue seen in T-30 .. T+60 on up to 3 earlier days keep p, others x 0.25). Variant "headev": the evidence =
the HEADS (queued, the downstream neighbour not queued) seen in T-60 .. T+60 on the same earlier days (the same hidden rows as seen()),
links within +-1 of a head keep p, others x 0.25 (strength unchanged, the radius and the anchor changed). Line (common): points > 0 on
M3 (judge) and January / February, combined z >= +1; the official June onset read and recorded. usage: TFB_M3VER=v3 py -B -X utf8 g356_onset_head_evidence.py <m3jf | jun80>"""
import os, sys, pickle
MODE = sys.argv[1]; R0 = "C:/Users/user/tfb/"; W = R0 + "work/t2/"
if MODE == "jun80":
    g = open(R0 + "review/g342b_jun80_run.py", encoding="utf-8").read(); g = g[:g.index("src = open(")]; sys.argv = ["g342b", "none"]; exec(compile(g, "g342b_head", "exec"))
sys.argv = sys.argv[:1]
s = open(R0 + "review/g170_onset_instrument.py", encoding="utf-8").read()


def sub(a, b, n=1):
    global s
    assert s.count(a) == n, (a[:70], s.count(a))
    s = s.replace(a, b)


HEADEV = '''

def seen_heads(c, t, k, lo=-12, hi=12, cap=3):
    hh = np.zeros(c.n, bool); nd = 0; hid = sim_blank(c.key, t // 288, min(cap, k), t)
    for back in range(1, min(cap, k) + 1):
        dd = t - back * 288
        if dd + lo < 0:
            break
        rr = [r for r in range(dd + lo, dd + hi) if r not in hid]
        if rr:
            qq = c.q[rr]; hh |= (qq & ~np.concatenate([qq[:, 1:], np.zeros((len(rr), 1), bool)], axis=1)).any(axis=0)
        nd += 1
    return hh, nd

'''
sub('def sets(c, p_, ph, a, t, k, a2=None):', HEADEV + '\ndef sets(c, p_, ph, a, t, k, a2=None):')
sub('    return {v: decode(x)[-1] for v, x in pm.items()}',
    '    hh_, ndh = seen_heads(c, t, k); xh = pa.copy()\n'
    '    if ndh:\n'
    '        xh[-1] = xh[-1] * np.where(G.dil(hh_, 1), 1.0, 0.25)\n'
    '    pm["headev"] = xh\n'
    '    return {v: decode(x)[-1] for v, x in pm.items()}')
sub('"hid", "hidf", "ac"); ND = 8', '"hid", "hidf", "ac", "headev"); ND = 8')
sub('            S = sets(CTX[key], PN[w], ph, a, m["t"], k, PO.adaptive_a(ev2, m["day"]))\n',
    '            S = sets(CTX[key], 0.8 * PN[w] + 0.2 * sum(PC_[j][w] for j in range(4)) / 4, ph, a, m["t"], k, PO.adaptive_a(ev2, m["day"]))\n')
sub('def march(limit=None):', 'PC_ = [pickle.load(open(W + f"probs_m3v3cap__pcL_s{s_}_fold2030-12.pkl", "rb")) for s_ in range(4)]\n\n\ndef march(limit=None):')
if MODE == "jun80":
    sub('p_ = (OOF[0][k] + OOF[1][k]) / 2;', 'p_ = (CV_[0][str(k)] + CV_[1][str(k)]) / 2;')
    sub('    for mo in MONTHS:\n        mdl = phys_lomo', '    for mo in ("2030-06",):\n        mdl = phys_lomo')
    sub('def march(limit=None):', 'CV_ = [pickle.load(open(W + f"probs_jun80__cvf_attnglob_s{s_}_fold2030-06.pkl", "rb")) for s_ in (0, 1)]\n\n\ndef march(limit=None):')
    sub("t0 = TM(); Dm = march(lim)", "t0 = TM(); Dm = None")
    sub('    print(f"M3 dev {len(Dm)} windows {TM() - t0:.0f}s | identity against g108 A: {np.abs(o - Dm.base.to_numpy()).max():.2e}", flush=True); ', '    ')
    sub("o = pd.read_csv(W + \"g108_rectangles.csv\").set_index(\"w\").A.reindex(Dm.w).to_numpy()", "o = None")
else:
    sub("o = pd.read_csv(W + \"g108_rectangles.csv\").set_index(\"w\").A.reindex(Dm.w).to_numpy()", "o = Dm.base.to_numpy()")
sub("t0 = TM(); Dt = train_months(lim)", f'''t0 = TM(); Dt = train_months(lim); Dt.to_csv(W + "g356_{MODE}_t.csv", index=False)
    cl = pd.factorize(Dt.panel + "_" + Dt.day.astype(str))[0]; Rm = np.random.default_rng(0).multinomial(cl.max() + 1, np.full(cl.max() + 1, 1 / (cl.max() + 1)), size=2000)
    x = (Dt.L_headev - Dt.L_base).to_numpy(); sej = float(((Rm @ np.bincount(cl, x)) / (Rm @ np.bincount(cl))).std()); zj = x.mean() / max(sej, 1e-12)
    lab_ = "June official onset" if Dm is None else "Jan / Feb onset"; print(f"{{lab_}}: base {{Dt.L_base.mean():.4f}}, headev - base {{x.mean():+.5f}} (SE {{sej:.5f}}, z {{zj:+.2f}}, changed {{int((np.abs(x) > 1e-12).sum())}})", flush=True)
    if Dm is not None:
        Dm.to_csv(W + "g356_m3.csv", index=False); sm = m3v2.summary(Dm, "base", "headev", "private"); zm = sm["t2_weighted"] / max(sm["se"], 1e-12)
        print(f"M3 onset: base {{Dm.base.mean():.4f}}, headev raw {{(Dm.headev - Dm.base).mean():+.5f}}, judge {{sm['t2_weighted']:+.5f}} (SE {{sm['se']:.5f}}, z {{zm:+.2f}}, changed {{int((Dm.headev != Dm.base).sum())}}) | combined z {{(zm + zj) / np.sqrt(2):+.2f}}; line: {{bool(sm['t2_weighted'] > 0 and x.mean() > 0 and (zm + zj) / np.sqrt(2) >= 1)}}", flush=True)
    sys.exit()
    Dt = None''')
exec(compile(s, "g170_g356", "exec"))
