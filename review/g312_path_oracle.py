"""G312 (11/05, ledger #247, measurement only): path oracles on the onset flow. G170's flow (M3 v3 dev onset 100 windows with A17's onset =
0.8 onset networks + 0.2 pcL s0-3, the physics onset adaptive a, the penalty 0.25; January / February 317 selector onset windows with G170's
train_months flow, L labels). Variants, all decoded as base:
  (i)  or<a> : the physics ranker's probability p_phys replaced by the true T+30 queue indicator (scored cells), mixed at a in {0.3, 0.5, 1.0}
  (ii) evor  : the evidence penalty lifted on the true sites only (the others x 0.25 as base)
Per variant: mean IoU, and the windows that move from a miss to a hit (k-expected IoU < 0.5 -> >= 0.5) and back.
usage: TFB_M3VER=v3 py -B -X utf8 g312_path_oracle.py"""
import sys, pickle
s = open(r"C:/Users/user/tfb/review/g170_onset_instrument.py", encoding="utf-8").read()
OV = ("or0.3", "or0.5", "or1.0", "evor")


def sub(a, b, n=1):
    global s
    assert s.count(a) == n, (a[:60], s.count(a))
    s = s.replace(a, b)


sub('"hid", "hidf", "ac"); ND = 8', '"hid", "hidf", "ac", ' + ", ".join(f'"{v}"' for v in OV) + '); ND = 8')
sub('    return {v: decode(x)[-1] for v, x in pm.items()}',
    '    yt_ = YT_[0].astype(float)\n'
    '    for a_ in (0.3, 0.5, 1.0):\n'
    '        q_ = p_.copy(); q_[5] = (1 - a_) * p_[5] + a_ * yt_; pm[f"or{a_}"] = rule(q_, 0.25)\n'
    '    x_ = pa.copy()\n'
    '    if nd:\n'
    '        x_[-1] = x_[-1] * np.where(ev | YT_[0], 1.0, 0.25)\n'
    '    pm["evor"] = x_\n'
    '    return {v: decode(x)[-1] for v, x in pm.items()}')
sub('            S = sets(CTX[key], PN[w], ph, a, m["t"], k, PO.adaptive_a(ev2, m["day"]))\n',
    '            YT_[0] = y & e; S = sets(CTX[key], 0.8 * PN[w] + 0.2 * sum(PC_[j][w] for j in range(4)) / 4, ph, a, m["t"], k, PO.adaptive_a(ev2, m["day"]))\n')
sub('                    S = sets(c, p_, ph, a, t, kk)\n', '                    YT_[0] = yL & eL; S = sets(c, p_, ph, a, t, kk)\n')
sub('def march(limit=None):', 'PC_ = [pickle.load(open(W + f"probs_m3v3cap__pcL_s{s_}_fold2030-12.pkl", "rb")) for s_ in range(4)]; YT_ = [None]\n\n\ndef march(limit=None):')
sub("t0 = TM(); Dm = march(lim)", "t0 = TM(); Dm = march(lim); P8 = pd.read_csv(W + 'g280_m3.csv').set_index('w'); print('identity base (A17 onset) against g280 pc4:', float(np.abs(Dm.set_index('w').base - P8.pc4.reindex(Dm.w).to_numpy()).max()), flush=True)")
sub("o = pd.read_csv(W + \"g108_rectangles.csv\").set_index(\"w\").A.reindex(Dm.w).to_numpy()", "o = Dm.base.to_numpy()")
sub("t0 = TM(); Dt = train_months(lim)", '''t0 = TM(); Dt = train_months(lim); Dm.to_csv(W + "g312_m3.csv", index=False); Dt.to_csv(W + "g312_jf.csv", index=False)
    for nm_, D_, b_ in (("M3 dev onset (A17)", Dm, "base"), ("Jan / Feb onset (OOF flow, L)", Dt, "L_base")):
        out = [f"base {D_[b_].mean():.4f}"]
        for v_ in ''' + str(OV) + ''':
            c_ = v_ if b_ == "base" else "L_" + v_; up = int(((D_[b_] < 0.5) & (D_[c_] >= 0.5)).sum()); dn = int(((D_[b_] >= 0.5) & (D_[c_] < 0.5)).sum())
            out.append(f"{v_} {D_[c_].mean():.4f} ({D_[c_].mean() - D_[b_].mean():+.4f}; miss->hit {up}, hit->miss {dn})")
        print(f"{nm_} {len(D_)} windows (misses at base {int((D_[b_] < 0.5).sum())}): " + " | ".join(out), flush=True)
    sys.exit()
    Dt = None''')
exec(compile(s, "g170p12", "exec"))
