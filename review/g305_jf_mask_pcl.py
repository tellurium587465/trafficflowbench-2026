"""G305 (11/05, ledger #244 V1, onset): does the M3-type history mask explain why pcL reads negative on M3 and positive on January / February?
The 317 January / February selector onset windows, G170's train_months flow (leave-one-month-out physics onset, adaptive a, the evidence
penalty 0.25 under the "mask" context, L labels), the onset networks = capB0 s0-3 (fold 2030-12, OOF for these months), mixed with pcL s0-3
(fold 2030-12) at 0.2 as A does. Inputs: G296 with the history unmasked (nn_test_jfonmh0, as the official windows) or masked as M3
(nn_test_jfonmh1, TFB_T2_MASKHIST=1), each through t2_pct for pcL. Read: pcL's effect (0.8 capB0 + 0.2 pcL - capB0) under mh0 and mh1, and
the paired difference mh1 - mh0 of that effect; cluster bootstrap (month x panel x day, 2000). For comparison, M3 dev's pcL effect on A17
(G280: -0.0063). usage: TFB_M3VER=v3 py -B -X utf8 g305_jf_mask_pcl.py"""
import sys, pickle
s = open(r"C:/Users/user/tfb/review/g170_onset_instrument.py", encoding="utf-8").read()
NM = ("c0", "c0p", "c1", "c1p")


def sub(a, b, n=1):
    global s
    assert s.count(a) == n, (a[:60], s.count(a))
    s = s.replace(a, b)


sub('"hid", "hidf", "ac"); ND = 8', '"hid", "hidf", "ac", ' + ", ".join(f'"{v}"' for v in NM) + '); ND = 8')
sub('    return {v: decode(x)[-1] for v, x in pm.items()}', '    out_ = {v: decode(x)[-1] for v, x in pm.items()}\n    for v_ in ' + str(NM) + ':\n        out_[v_] = out_["base"]\n    return out_')
sub('def march(limit=None):', 'L_ = lambda pre, tag: [pickle.load(open(W + f"probs_{pre}__{tag}_s{s_}_fold2030-12.pkl", "rb")) for s_ in range(4)]\n'
    'CB = {m_: L_(f"jfonmh{m_}x", "capB0") for m_ in "01"}; PL = {m_: L_(f"jfonmh{m_}p", "pcL") for m_ in "01"}\n\n\ndef march(limit=None):')
sub('                    S = sets(c, p_, ph, a, t, kk)\n', '                    S = sets(c, p_, ph, a, t, kk)\n'
    '                    for m_ in "01":\n'
    '                        cb_ = sum(d_[str(k)] for d_ in CB[m_]) / 4; pl_ = sum(d_[str(k)] for d_ in PL[m_]) / 4\n'
    '                        S[f"c{m_}"] = sets(c, cb_, ph, a, t, kk)["base"]; S[f"c{m_}p"] = sets(c, 0.8 * cb_ + 0.2 * pl_, ph, a, t, kk)["base"]\n')
sub("t0 = TM(); Dt = train_months(lim)", '''t0 = TM(); Dt = train_months(lim); Dt.to_csv(W + "g305_jf.csv", index=False); print(f"Jan / Feb onset {len(Dt)} windows ({TM() - t0:.0f}s)", flush=True)
    cl = pd.factorize(Dt.month + "_" + Dt.panel + "_" + Dt.day.astype(str))[0]; Rm = np.random.default_rng(0).multinomial(cl.max() + 1, np.full(cl.max() + 1, 1 / (cl.max() + 1)), size=2000)
    bse = lambda d_: float(((Rm @ np.bincount(cl, d_)) / (Rm @ np.bincount(cl))).std())
    e0 = (Dt.L_c0p - Dt.L_c0).to_numpy(); e1 = (Dt.L_c1p - Dt.L_c1).to_numpy(); dd = e1 - e0
    print(f"capB0 level: mh0 {Dt.L_c0.mean():.4f}, mh1 {Dt.L_c1.mean():.4f} (mask cost {(Dt.L_c1 - Dt.L_c0).mean():+.4f})", flush=True)
    print(f"pcL effect (0.8 capB0 + 0.2 pcL - capB0): mh0 (official-like) {e0.mean():+.5f} (SE {bse(e0):.5f}) | mh1 (M3-like) {e1.mean():+.5f} (SE {bse(e1):.5f}) | mh1 - mh0 {dd.mean():+.5f} (SE {bse(dd):.5f}, z {dd.mean() / max(bse(dd), 1e-12):+.2f}) | windows where the effect moves {int((np.abs(dd) > 1e-12).sum())}", flush=True)
    sys.exit()
    Dt = None''')
exec(compile(s, "g170p11", "exec"))
