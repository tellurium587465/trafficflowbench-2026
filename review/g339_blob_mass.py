"""G339 (11/05, ledger #273): onset decode ranked by blob mass. G170's flow (M3 v3 dev onset 100 windows with A17's onset = 0.8 q24f + 0.2 pcL
4 seeds as G335; January / February 317 onset windows with the OOF flow). After the penalty (b = rule(pa, 0.25)), on the T+30 row of b:
blobs = maximal runs of links with p > 0.05; chosen = the blob of the largest mass among those decode(b)[-1] touches; heaviest = the blob of
the largest mass. Where they differ: the T+30 row of b outside the heaviest blob -> 0 and decode again (other rows as they are); elsewhere
the base. Line (#273, common): points > 0 on both M3 (judge, private composition) and January / February (L), combined z >= +1, and April
rows change (checked separately). usage: TFB_M3VER=v3 py -B -X utf8 g339_blob_mass.py"""
import sys, pickle
s = open(r"C:/Users/user/tfb/review/g170_onset_instrument.py", encoding="utf-8").read()


def sub(a, b, n=1):
    global s
    assert s.count(a) == n, (a[:60], s.count(a))
    s = s.replace(a, b)


PRE = '''
NCH_ = [0, 0]


def blobs_(row, th=0.05):
    out, j = [], 0
    while j < len(row):
        if row[j] > th:
            e_ = j
            while e_ + 1 < len(row) and row[e_ + 1] > th:
                e_ += 1
            out.append((j, e_)); j = e_ + 1
        else:
            j += 1
    return out


def blobfix(x):
    S = decode(x); r = x[-1]; bl = blobs_(r); NCH_[1] += 1
    if not bl:
        return S
    ms = [float(r[a:b + 1].sum()) for a, b in bl]; ch = [i for i, (a, b) in enumerate(bl) if S[-1][a:b + 1].any()]
    if not ch:
        return S
    ci = max(ch, key=lambda i: ms[i]); mi = int(np.argmax(ms))
    if mi == ci:
        return S
    a, b = bl[mi]; m_ = np.zeros(len(r), bool); m_[a:b + 1] = True; x2 = x.copy(); x2[-1] = np.where(m_, r, 0.0); NCH_[0] += 1; return decode(x2)

'''
sub('def sets(c, p_, ph, a, t, k, a2=None):', PRE + '\ndef sets(c, p_, ph, a, t, k, a2=None):')
sub('"hid", "hidf", "ac"); ND = 8', '"hid", "hidf", "ac", "blob"); ND = 8')
sub('    return {v: decode(x)[-1] for v, x in pm.items()}', '    out_ = {v: decode(x)[-1] for v, x in pm.items()}; out_["blob"] = blobfix(b)[-1]; return out_')
sub('            S = sets(CTX[key], PN[w], ph, a, m["t"], k, PO.adaptive_a(ev2, m["day"]))\n',
    '            S = sets(CTX[key], 0.8 * PN[w] + 0.2 * sum(PC_[j][w] for j in range(4)) / 4, ph, a, m["t"], k, PO.adaptive_a(ev2, m["day"]))\n')
sub('def march(limit=None):', 'PC_ = [pickle.load(open(W + f"probs_m3v3cap__pcL_s{s_}_fold2030-12.pkl", "rb")) for s_ in range(4)]\n\n\ndef march(limit=None):')
sub("t0 = TM(); Dm = march(lim)", "t0 = TM(); Dm = march(lim); P8 = pd.read_csv(W + 'g280_m3.csv').set_index('w'); print('identity base (A17 onset) against g280 pc4:', float(np.abs(Dm.set_index('w').base - P8.pc4.reindex(Dm.w).to_numpy()).max()), '| (window, k) re-decoded', NCH_, flush=True); NCH_[:] = [0, 0]")
sub("o = pd.read_csv(W + \"g108_rectangles.csv\").set_index(\"w\").A.reindex(Dm.w).to_numpy()", "o = Dm.base.to_numpy()")
sub("t0 = TM(); Dt = train_months(lim)", '''t0 = TM(); Dt = train_months(lim); print("Jan / Feb (window, k) re-decoded", NCH_, flush=True)
    Dm.to_csv(W + "g339_m3.csv", index=False); Dt.to_csv(W + "g339_jf.csv", index=False)
    s_ = m3v2.summary(Dm, "base", "blob", "private"); zm = s_["t2_weighted"] / max(s_["se"], 1e-9); lm, ls = ratio(Dt, (Dt.L_blob - Dt.L_base).to_numpy(), np.ones(len(Dt))); zj = lm / max(ls, 1e-9)
    chm = int((Dm.blob != Dm.base).sum()); chj = int((Dt.L_blob != Dt.L_base).sum())
    print(f"M3: raw {(Dm.blob - Dm.base).mean():+.4f}, judge {s_['t2_weighted']:+.4f} (SE {s_['se']:.4f}, z {zm:+.2f}), windows changed {chm} | Jan / Feb L {lm:+.4f} (SE {ls:.4f}, z {zj:+.2f}), windows changed {chj} | combined z {(zm + zj) / np.sqrt(2):+.2f}", flush=True)
    print("line (both > 0 and combined z >= +1):", bool(s_["t2_weighted"] > 0 and lm > 0 and (zm + zj) / np.sqrt(2) >= 1), flush=True)
    sys.exit()
    Dt = None''')
exec(compile(s, "g170p339", "exec"))
