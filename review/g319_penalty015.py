"""G319 (11/05, ledger #255): the evidence penalty's lower end, 0.25 -> 0.15, on A17 / A21's onset (identical onset). G170's flow: M3 v3 dev
onset 100 windows with A17's onset (pc4) and the 317 January / February onset windows (OOF flow, L labels). Line: M3 point > 0 and z >= +1,
January / February z >= -1.64 (with the LB pair >= 0 and April rows changing, checked separately). usage: TFB_M3VER=v3 py -B -X utf8 g319_penalty015.py"""
import sys, pickle
s = open(r"C:/Users/user/tfb/review/g170_onset_instrument.py", encoding="utf-8").read()


def sub(a, b, n=1):
    global s
    assert s.count(a) == n, (a[:60], s.count(a))
    s = s.replace(a, b)


L1 = '            S = sets(CTX[key], PN[w], ph, a, m["t"], k, PO.adaptive_a(ev2, m["day"]))\n'
sub('"hid", "hidf", "ac"); ND = 8', '"hid", "hidf", "ac", "B0.15"); ND = 8')
sub('"B0.35": rule(pa, 0.35),', '"B0.35": rule(pa, 0.35), "B0.15": rule(pa, 0.15),')
sub(L1, '            S = sets(CTX[key], 0.8 * PN[w] + 0.2 * sum(PC_[j][w] for j in range(4)) / 4, ph, a, m["t"], k, PO.adaptive_a(ev2, m["day"]))\n')
sub('def march(limit=None):', 'PC_ = [pickle.load(open(W + f"probs_m3v3cap__pcL_s{s_}_fold2030-12.pkl", "rb")) for s_ in range(4)]\n\n\ndef march(limit=None):')
sub("t0 = TM(); Dm = march(lim)", "t0 = TM(); Dm = march(lim); P8 = pd.read_csv(W + 'g280_m3.csv').set_index('w'); print('identity base (A17 onset) against g280 pc4:', float(np.abs(Dm.set_index('w').base - P8.pc4.reindex(Dm.w).to_numpy()).max()), flush=True)")
sub("o = pd.read_csv(W + \"g108_rectangles.csv\").set_index(\"w\").A.reindex(Dm.w).to_numpy()", "o = Dm.base.to_numpy()")
sub("t0 = TM(); Dt = train_months(lim)", '''t0 = TM(); Dt = train_months(lim); Dm.to_csv(W + "g319_m3.csv", index=False); Dt.to_csv(W + "g319_jf.csv", index=False)
    r1 = m3v2.summary(Dm, "base", "B0.15", "private"); a1, s1 = r1["t2_weighted"], r1["se"]
    cl = pd.factorize(Dt.month + "_" + Dt.panel + "_" + Dt.day.astype(str))[0]; Rm = np.random.default_rng(0).multinomial(cl.max() + 1, np.full(cl.max() + 1, 1 / (cl.max() + 1)), size=2000)
    d2 = (Dt["L_B0.15"] - Dt.L_base).to_numpy(); a2 = d2.mean(); s2 = float(((Rm @ np.bincount(cl, d2)) / (Rm @ np.bincount(cl))).std())
    print(f"penalty 0.15 - 0.25: M3 dev onset (A17) {a1:+.5f} (SE {s1:.5f}, z {a1 / max(s1, 1e-12):+.2f}, changed {int((~np.isclose(Dm['B0.15'], Dm.base)).sum())}) | Jan / Feb onset (L) {a2:+.5f} (SE {s2:.5f}, z {a2 / max(s2, 1e-12):+.2f}, changed {int((np.abs(d2) > 1e-12).sum())}) | instrument part of the line: {a1 > 0 and a1 / max(s1, 1e-12) >= 1 and a2 / max(s2, 1e-12) >= -1.64}", flush=True)
    sys.exit()
    Dt = None''')
exec(compile(s, "g170p17", "exec"))
