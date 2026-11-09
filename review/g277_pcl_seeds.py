"""G277 (11/05, ledger #232 measured before any GPU training): the rank-input onset network's seed count. G170's onset instrument on
M3 v3 dev (100 windows, hold-out days not read), final A's onset flow with the rank-input network mixed at 0.2 into the onset networks'
probabilities (as #17 / A12), the M3 probabilities of the fold-2 versions pcL s0, s1 (the only ones on M3): both seeds against one seed;
d = IoU(2) - mean IoU(1), the c / n law predicts 2 -> 4 as d / 2. usage: TFB_M3VER=v3 py -B -X utf8 g277_pcl_seeds.py"""
import sys, pickle
s = open(r"C:/Users/user/tfb/review/g170_onset_instrument.py", encoding="utf-8").read()


def sub(a, b, n=1):
    global s
    assert s.count(a) == n, (a[:60], s.count(a))
    s = s.replace(a, b)


sub('"hid", "hidf", "ac"); ND = 8', '"hid", "hidf", "ac", "pc2", "pc0", "pc1"); ND = 8')
sub('            S = sets(CTX[key], PN[w], ph, a, m["t"], k, PO.adaptive_a(ev2, m["day"]))\n            for v in VARS:\n                sc[v] += pr * iou1(S[v], y, e)\n',
    '            S = sets(CTX[key], PN[w], ph, a, m["t"], k, PO.adaptive_a(ev2, m["day"]))\n            for v in VARS[:-3]:\n                sc[v] += pr * iou1(S[v], y, e)\n'
    '            for v_, pc_ in (("pc2", (PC_[0][w] + PC_[1][w]) / 2), ("pc0", PC_[0][w]), ("pc1", PC_[1][w])):\n'
    '                sc[v_] += pr * iou1(sets(CTX[key], 0.8 * PN[w] + 0.2 * pc_, ph, a, m["t"], k, PO.adaptive_a(ev2, m["day"]))["base"], y, e)\n')
sub('def march(limit=None):', 'PC_ = [pickle.load(open(W + f"probs_m3v3cap__pcL_s{s_}_fold2030-12.pkl", "rb")) for s_ in (0, 1)]\n\n\ndef march(limit=None):')
sub("t0 = TM(); Dt = train_months(lim)",
    'Dm.to_csv(W + "g277_m3.csv", index=False); d_ = (Dm.pc2 - (Dm.pc0 + Dm.pc1) / 2).to_numpy(); rg_ = np.random.default_rng(0); bs_ = [d_[rg_.integers(0, len(d_), len(d_))].mean() for _ in range(300)]; '
    'print(f"pcL 1 -> 2 seeds (M3 dev onset {len(Dm)}): d {d_.mean():+.5f} (SE {np.std(bs_):.5f}) -> predicted 2 -> 4 {d_.mean() / 2:+.5f} | pc2 vs base (no pcL) {(Dm.pc2 - Dm.base).mean():+.4f} | windows where a seed differs {int((~np.isclose(Dm.pc0, Dm.pc2) | ~np.isclose(Dm.pc1, Dm.pc2)).sum())}", flush=True); sys.exit()')
exec(compile(s, "g170p2", "exec"))
