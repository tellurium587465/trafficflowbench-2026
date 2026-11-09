"""G278 (11/05, ledger #231, declared 11/04 23:22 before the result): onset over q24f seeds 0-15 (80 networks) against 0-7 (40), on
M3 v3 dev onset 100 windows (hold-out days not read), G170's flow (final A's onset, without pcL as in G170's base); line: 16 - 8 >= -1 SE.
Patched from G170 as G277. usage: TFB_M3VER=v3 py -B -X utf8 g278_onset16.py"""
import sys, pickle
s = open(r"C:/Users/user/tfb/review/g170_onset_instrument.py", encoding="utf-8").read()


def sub(a, b, n=1):
    global s
    assert s.count(a) == n, (a[:60], s.count(a))
    s = s.replace(a, b)


L1 = '            S = sets(CTX[key], PN[w], ph, a, m["t"], k, PO.adaptive_a(ev2, m["day"]))\n'
sub('"hid", "hidf", "ac"); ND = 8', '"hid", "hidf", "ac", "s16"); ND = 8')
sub(L1 + '            for v in VARS:\n                sc[v] += pr * iou1(S[v], y, e)\n',
    L1 + '            for v in VARS[:-1]:\n                sc[v] += pr * iou1(S[v], y, e)\n'
    '            sc["s16"] += pr * iou1(sets(CTX[key], PN16[w], ph, a, m["t"], k, PO.adaptive_a(ev2, m["day"]))["base"], y, e)\n')
sub('def march(limit=None):', '''def pn16():
    out = {}
    for s_ in range(16):
        for c_, wt in g2.CF5:
            P_ = pickle.load(open(W + f"probs_{G.PRE}__q24f_{c_}_s{s_}.pkl", "rb"))
            for w_ in g2.DEV:
                out[w_] = out.get(w_, 0) + wt * P_[w_] / (8.0 * 16)
    return out


PN16 = pn16()


def march(limit=None):''')
sub("t0 = TM(); Dt = train_months(lim)",
    'Dm.to_csv(W + "g278_m3.csv", index=False); d_ = (Dm.s16 - Dm.base).to_numpy(); rg_ = np.random.default_rng(0); bs_ = [d_[rg_.integers(0, len(d_), len(d_))].mean() for _ in range(2000)]; '
    'print(f"#231 onset 16 - 8 seeds (M3 dev onset {len(Dm)}, PRE {G.PRE}): {d_.mean():+.5f} (SE {np.std(bs_):.5f}, z {d_.mean() / np.std(bs_):+.2f}) | base {Dm.base.mean():.4f} s16 {Dm.s16.mean():.4f} | windows that differ {int((~np.isclose(Dm.s16, Dm.base)).sum())} | line >= -1 SE: {d_.mean() >= -np.std(bs_)}", flush=True); sys.exit()')
exec(compile(s, "g170p3", "exec"))
