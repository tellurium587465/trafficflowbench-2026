"""G316 (11/05, ledger #250, onset): the site masses as mutually exclusive blobs. On the T+30 row of the probabilities just before best_set
(G170's flow: M3 v3 dev onset 100 windows with A17's onset, January / February 317 onset windows with the OOF flow, L labels), the blobs are
the maximal runs of links with p > 0.05; the decode picks ONE contiguous range [a, b] inside one blob (or nothing) that maximises
A / (|S| + P - A) (best_set's approximation of the expected IoU; P = the row's total mass) instead of best_set's cell-wise set.
Line (#247-#252): both points > 0 and merged z >= +1. usage: TFB_M3VER=v3 py -B -X utf8 g316_onset_blob_decode.py"""
import sys, pickle
s = open(r"C:/Users/user/tfb/review/g170_onset_instrument.py", encoding="utf-8").read()


def sub(a, b, n=1):
    global s
    assert s.count(a) == n, (a[:60], s.count(a))
    s = s.replace(a, b)


BLOB = '''

def blob_row(r, th=0.05):
    n = len(r); P = float(r.sum()); best, arg = 0.0, None; j = 0
    while j < n:
        if r[j] > th:
            e = j
            while e + 1 < n and r[e + 1] > th:
                e += 1
            pre = np.concatenate([[0.0], np.cumsum(r[j:e + 1])])
            for a in range(e - j + 1):
                for b in range(a, e - j + 1):
                    A = float(pre[b + 1] - pre[a]); sc = A / ((b - a + 1) + P - A)
                    if sc > best:
                        best, arg = sc, (j + a, j + b)
            j = e + 1
        else:
            j += 1
    out = np.zeros(n, bool)
    if arg:
        out[arg[0]:arg[1] + 1] = True
    return out

'''
sub('def sets(c, p_, ph, a, t, k, a2=None):', BLOB + '\ndef sets(c, p_, ph, a, t, k, a2=None):')
sub('"hid", "hidf", "ac"); ND = 8', '"hid", "hidf", "ac", "blob"); ND = 8')
sub('    return {v: decode(x)[-1] for v, x in pm.items()}', '    out_ = {v: decode(x)[-1] for v, x in pm.items() if v != "blob"}; out_["blob"] = blob_row(b[-1]); return out_')
sub('            S = sets(CTX[key], PN[w], ph, a, m["t"], k, PO.adaptive_a(ev2, m["day"]))\n',
    '            S = sets(CTX[key], 0.8 * PN[w] + 0.2 * sum(PC_[j][w] for j in range(4)) / 4, ph, a, m["t"], k, PO.adaptive_a(ev2, m["day"]))\n')
sub('def march(limit=None):', 'PC_ = [pickle.load(open(W + f"probs_m3v3cap__pcL_s{s_}_fold2030-12.pkl", "rb")) for s_ in range(4)]\n\n\ndef march(limit=None):')
sub("t0 = TM(); Dm = march(lim)", "t0 = TM(); Dm = march(lim); P8 = pd.read_csv(W + 'g280_m3.csv').set_index('w'); print('identity base (A17 onset) against g280 pc4:', float(np.abs(Dm.set_index('w').base - P8.pc4.reindex(Dm.w).to_numpy()).max()), flush=True)")
sub("o = pd.read_csv(W + \"g108_rectangles.csv\").set_index(\"w\").A.reindex(Dm.w).to_numpy()", "o = Dm.base.to_numpy()")
sub("t0 = TM(); Dt = train_months(lim)", '''t0 = TM(); Dt = train_months(lim); Dm.to_csv(W + "g316_m3.csv", index=False); Dt.to_csv(W + "g316_jf.csv", index=False)
    r1 = m3v2.summary(Dm, "base", "blob", "private"); a1, s1 = r1["t2_weighted"], r1["se"]
    cl = pd.factorize(Dt.month + "_" + Dt.panel + "_" + Dt.day.astype(str))[0]; Rm = np.random.default_rng(0).multinomial(cl.max() + 1, np.full(cl.max() + 1, 1 / (cl.max() + 1)), size=2000)
    d2 = (Dt.L_blob - Dt.L_base).to_numpy(); a2 = d2.mean(); s2 = float(((Rm @ np.bincount(cl, d2)) / (Rm @ np.bincount(cl))).std())
    w1, w2 = 1 / max(s1, 1e-9) ** 2, 1 / max(s2, 1e-9) ** 2; mg = (w1 * a1 + w2 * a2) / (w1 + w2); sm = (w1 + w2) ** -0.5
    print(f"M3 dev onset (A17): blob - best_set {a1:+.5f} (SE {s1:.5f}, z {a1 / max(s1, 1e-12):+.2f}, changed {int((~np.isclose(Dm.blob, Dm.base)).sum())}) | Jan / Feb onset (L): {a2:+.5f} (SE {s2:.5f}, z {a2 / max(s2, 1e-12):+.2f}, changed {int((np.abs(d2) > 1e-12).sum())}) | merged {mg:+.5f} (z {mg / sm:+.2f}) | line: {a1 > 0 and a2 > 0 and mg / sm >= 1}", flush=True)
    sys.exit()
    Dt = None''')
exec(compile(s, "g170p14", "exec"))
