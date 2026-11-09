"""G280 (11/05, ledger #232, strategy 11/05: the harmlessness check before A16): the rank-input onset network over seeds 0-3 against 0-1,
mixed at 0.2 into the onset networks' probabilities just before the physics onset (weight unchanged), G170's flow on M3 v3 dev onset 100
windows (hold-out days not read); the M3 probabilities of the fold-2 networks pcL s0-s3 (the production files). Line: 4 - 2 >= -1 SE.
Also printed: 4 seeds and 2 seeds against no pcL. Patched from G170 as G277. usage: TFB_M3VER=v3 py -B -X utf8 g280_pcl4.py"""
import sys, pickle
s = open(r"C:/Users/user/tfb/review/g170_onset_instrument.py", encoding="utf-8").read()


def sub(a, b, n=1):
    global s
    assert s.count(a) == n, (a[:60], s.count(a))
    s = s.replace(a, b)


L1 = '            S = sets(CTX[key], PN[w], ph, a, m["t"], k, PO.adaptive_a(ev2, m["day"]))\n'
sub('"hid", "hidf", "ac"); ND = 8', '"hid", "hidf", "ac", "pc2", "pc4"); ND = 8')
sub(L1 + '            for v in VARS:\n                sc[v] += pr * iou1(S[v], y, e)\n',
    L1 + '            for v in VARS[:-2]:\n                sc[v] += pr * iou1(S[v], y, e)\n'
    '            for v_, pc_ in (("pc2", sum(PC_[j][w] for j in range(2)) / 2), ("pc4", sum(PC_[j][w] for j in range(4)) / 4)):\n'
    '                sc[v_] += pr * iou1(sets(CTX[key], 0.8 * PN[w] + 0.2 * pc_, ph, a, m["t"], k, PO.adaptive_a(ev2, m["day"]))["base"], y, e)\n')
sub('def march(limit=None):', 'PC_ = [pickle.load(open(W + f"probs_m3v3cap__pcL_s{s_}_fold2030-12.pkl", "rb")) for s_ in range(4)]\n\n\ndef march(limit=None):')
sub("t0 = TM(); Dt = train_months(lim)",
    'Dm.to_csv(W + "g280_m3.csv", index=False); rg_ = np.random.default_rng(0); ix_ = [rg_.integers(0, len(Dm), len(Dm)) for _ in range(2000)]\n'
    '    for a_, b_ in (("pc4", "pc2"), ("pc4", "base"), ("pc2", "base")):\n'
    '        d_ = (Dm[a_] - Dm[b_]).to_numpy(); se_ = float(np.std([d_[i_].mean() for i_ in ix_]))\n'
    '        print(f"{a_} - {b_} (M3 dev onset {len(Dm)}): {d_.mean():+.5f} (SE {se_:.5f}, z {d_.mean() / max(se_, 1e-12):+.2f}) | windows that differ {int((~np.isclose(Dm[a_], Dm[b_])).sum())}" + (f" | line >= -1 SE: {d_.mean() >= -se_}" if b_ == "pc2" else ""), flush=True)\n'
    '    sys.exit()\n    Dt = None')
exec(compile(s, "g170p4", "exec"))
