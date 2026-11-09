"""G335 (11/05, ledger #266, measurement only): month-map oracles for the physics term of onset, and the information curve f(k). G170's
flow (M3 v3 dev onset 100 windows with A17's onset; January / February 317 onset windows with the OOF flow; mode jun: the June 2030 replica
selector onset windows (#254's 330, not the official 80) with cvf fold 2030-06). p_phys replaced by
  tru  : the true T+30 queue indicator (as G312)
  map  : per link, the share of the month's OTHER days (the window's day left out, all days) with a head onset (queued after >= 12 visible
         free rows, as G314, and its downstream neighbour not queued) at that link within +-1 h of the window's T+30 clock (oracle: uses later days)
  prev : the same from the days before the window's day only (legal), reported by the number of those days k
each at a = the window's adaptive a (today) and a = 1.0; then the penalty 0.25 and best_set as base. Per variant: mean onset IoU, windows
moving miss -> hit (< 0.5 -> >= 0.5) and back; prev also by k in {0, 1, 2-3, 4-7, 8-14, 15+}.
usage: TFB_M3VER=v3 py -B -X utf8 g335_month_map_oracle.py [m3jf | jun]"""
import sys, pickle
MODE = (sys.argv[1:] or ["m3jf"])[0]; sys.argv = sys.argv[:1]
s = open(r"C:/Users/user/tfb/review/g170_onset_instrument.py", encoding="utf-8").read()
VV = ("tru_a", "tru_1", "map_a", "map_1", "prev_a", "prev_1")


def sub(a, b, n=1):
    global s
    assert s.count(a) == n, (a[:60], s.count(a))
    s = s.replace(a, b)


PRE = '''
ONM = {}


def onset_matrix(key, V, vcut):
    if key not in ONM:
        fin = np.isfinite(V); qd = fin & (V <= vcut[None, :]); fr = fin & ~qd; free12 = np.zeros_like(fr); run = np.zeros(V.shape[1], int)
        for r in range(len(V)):
            free12[r] = run >= 12; run = np.where(fr[r], run + 1, np.where(fin[r], 0, run))
        hd = qd & ~np.concatenate([qd[:, 1:], np.zeros((len(qd), 1), bool)], axis=1); ONM[key] = qd & free12 & hd   # heads only (the downstream neighbour not queued)
    return ONM[key]


def map_rates(c, t):
    on = onset_matrix(c.key, c.ns.V, VC[c.key]); r0, r1 = MS_[0]; t30 = t + 6; day = (t - r0) // 288; nd = (r1 - r0) // 288; tod = t30 % 288
    def rate(days):
        if not days:
            return np.zeros(c.n)
        hit = np.zeros(c.n)
        for d in days:
            a_, b_ = r0 + d * 288 + max(tod - 12, 0), r0 + d * 288 + min(tod + 13, 288)
            hit += on[a_:b_].any(axis=0)
        return hit / len(days)
    KD_[0] = day; return rate([d for d in range(nd) if d != day]), rate(list(range(day)))

'''
sub('def sets(c, p_, ph, a, t, k, a2=None):', PRE + '\ndef sets(c, p_, ph, a, t, k, a2=None):')
sub('"hid", "hidf", "ac"); ND = 8', '"hid", "hidf", "ac", ' + ", ".join(f'"{v}"' for v in VV) + '); ND = 8')
sub('    return {v: decode(x)[-1] for v, x in pm.items()}',
    '    mr, pv = map_rates(c, t); yt_ = YT_[0].astype(float)\n'
    '    for nm_, vec in (("tru", yt_), ("map", mr), ("prev", pv)):\n'
    '        for tag_, a_ in (("a", a), ("1", 1.0)):\n'
    '            q_ = p_.copy(); q_[5] = (1 - a_) * p_[5] + a_ * vec; pm[f"{nm_}_{tag_}"] = rule(q_, 0.25)\n'
    '    return {v: decode(x)[-1] for v, x in pm.items()}')
sub('            S = sets(CTX[key], PN[w], ph, a, m["t"], k, PO.adaptive_a(ev2, m["day"]))\n',
    '            YT_[0] = y & e; MS_[0] = (0, CTX[key].ns.V.shape[0]); S = sets(CTX[key], 0.8 * PN[w] + 0.2 * sum(PC_[j][w] for j in range(4)) / 4, ph, a, m["t"], k, PO.adaptive_a(ev2, m["day"]))\n')
sub('            CTX[key] = Ctx(key, P, P.V)\n', '            CTX[key] = Ctx(key, P, P.V); VC[key] = P.vcut\n')
sub('CTX[key] = Ctx(key, P, np.where(P.blank, np.nan, P.V), P.V); c = CTX[key];', 'CTX[key] = Ctx(key, P, np.where(P.blank, np.nan, P.V), P.V); c = CTX[key]; VC[key] = P.vcut; _mr = np.flatnonzero([s_.startswith(mo) for s_ in P.times]); MS_[0] = (int(_mr[0]), int(_mr[-1]) + 1);')
sub('                    S = sets(c, p_, ph, a, t, kk)\n', '                    YT_[0] = yL & eL; S = sets(c, p_, ph, a, t, kk); KDS_[int(k)] = KD_[0]\n')
sub('def march(limit=None):', 'PC_ = [pickle.load(open(W + f"probs_m3v3cap__pcL_s{s_}_fold2030-12.pkl", "rb")) for s_ in range(4)]; YT_ = [None]; MS_ = [None]; KD_ = [0]; KDS_ = {}; VC = {}\n\n\ndef march(limit=None):')
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
sub("t0 = TM(); Dt = train_months(lim)", '''t0 = TM(); Dt = train_months(lim); Dt["kdays"] = Dt.w.map(KDS_)
    def rep(D_, b_, lab):
        out = [f"base {D_[b_].mean():.4f}"]
        for v_ in ''' + str(VV) + ''':
            c_ = v_ if b_ == "base" else "L_" + v_; up = int(((D_[b_] < 0.5) & (D_[c_] >= 0.5)).sum()); dn = int(((D_[b_] >= 0.5) & (D_[c_] < 0.5)).sum())
            out.append(f"{v_} {D_[c_].mean() - D_[b_].mean():+.4f} (miss->hit {up}, hit->miss {dn})")
        print(f"{lab} {len(D_)} windows (misses at base {int((D_[b_] < 0.5).sum())}): " + " | ".join(out), flush=True)
        kk = D_["day"] if "kdays" not in D_ or D_["kdays"].isna().all() else D_["kdays"]
        for lo_, hi_ in ((0, 0), (1, 1), (2, 3), (4, 7), (8, 14), (15, 99)):
            mm = (kk >= lo_) & (kk <= hi_)
            if mm.any():
                print(f"   k {lo_}-{hi_}: {int(mm.sum())} windows, prev_a {(D_[('' if b_ == 'base' else 'L_') + 'prev_a'] - D_[b_])[mm].mean():+.4f}, prev_1 {(D_[('' if b_ == 'base' else 'L_') + 'prev_1'] - D_[b_])[mm].mean():+.4f}, map_1 {(D_[('' if b_ == 'base' else 'L_') + 'map_1'] - D_[b_])[mm].mean():+.4f}", flush=True)
    if Dm is not None:
        Dm.to_csv(W + "g335_m3.csv", index=False); rep(Dm, "base", "M3 dev onset (A17)")
    Dt.to_csv(W + f"g335_{'jun' if Dm is None else 'jf'}.csv", index=False); rep(Dt, "L_base", "June replica onset (cvf)" if Dm is None else "Jan / Feb onset (OOF flow)")
    sys.exit()
    Dt = None''')
exec(compile(s, "g170p20", "exec"))
