"""G252 read (11/04, ledger #217): G170's onset instrument with two arms more, decode only: s30 / s50 = the physics onset term ph
replaced by (1 - w) ph + w ps, w = 0.3 / 0.5, where ps = G144's site softmax (G252's refits) times the window's sum of ph (the same
total, redistributed over the links by the site model); the evidence rule and the decode as the finals. January / February: the model
with that month left out; M3 dev (100 windows, hold-out days not read): the nine-month model. Readings as G170 (L, F, M; judge).
usage: TFB_M3VER=v3 py -B -X utf8 g252_site_read.py [time | run]"""
import sys
s = open(r"C:/Users/user/tfb/review/g170_onset_instrument.py", encoding="utf-8").read()


def sub(a, b, n=1):
    global s
    assert s.count(a) == n, (a[:60], s.count(a))
    s = s.replace(a, b)


sub('"hid", "hidf", "ac"); ND = 8', '"hid", "hidf", "ac", "s30", "s50"); ND = 8')
sub('def sets(c, p_, ph, a, t, k, a2=None):', 'def sets(c, p_, ph, a, t, k, a2=None, ps=None):')
sub('    return {v: decode(x)[-1] for v, x in pm.items()}\n',
    '''    for wv, nm_ in ((0.3, "s30"), (0.5, "s50")):
        if ps is None:
            pm[nm_] = b
        else:
            q_ = p_.copy(); q_[5] = (1 - a) * p_[5] + a * ((1 - wv) * ph + wv * ps); pm[nm_] = rule(q_, 0.25)
    return {v: decode(x)[-1] for v, x in pm.items()}
''')
sub('def march(limit=None):', '''import torch as _t
NS_ = {"__name__": "g144"}; _sa = sys.argv; sys.argv = ["C:/Users/user/tfb/review/g144_onset_place.py", "run"]
exec(compile(open("C:/Users/user/tfb/review/g144_onset_place.py", encoding="utf-8").read().split("rows = load_train()")[0], "g144", "exec"), NS_); sys.argv = _sa
NETS = {}
for h_ in ("all", "2031-01", "2031-02"):
    ns_ = []
    for d_ in _t.load("C:/Users/user/tfb/work/t2/onset_place/g252_" + h_ + ".pt"):
        n_ = NS_["Net"](d_["i.weight"].shape[1]); n_.load_state_dict(d_); n_.eval(); ns_.append(n_)
    NETS[h_] = ns_


def SC_(nets, Xw):
    return NS_["score"](nets, NS_["inputs"](np.asarray(Xw)))


def march(limit=None):''')
sub('ph = phys.predict(PO.feats(Xw)); a = PO.adaptive_a(ev, m["day"]);', 'ph = phys.predict(PO.feats(Xw)); PS_ = SC_(NETS["all"], Xw) * ph.sum(); a = PO.adaptive_a(ev, m["day"]);')
sub('            S = sets(CTX[key], PN[w], ph, a, m["t"], k, PO.adaptive_a(ev2, m["day"]))', '            S = sets(CTX[key], PN[w], ph, a, m["t"], k, PO.adaptive_a(ev2, m["day"]), ps=PS_)')
sub('ph = mdl.predict(PO.feats(Xw)); a = PO.adaptive_a(ev, dom)', 'ph = mdl.predict(PO.feats(Xw)); PS_ = SC_(NETS[mo], Xw) * ph.sum(); a = PO.adaptive_a(ev, dom)')
sub('S = sets(c, p_, ph, a, t, kk)', 'S = sets(c, p_, ph, a, t, kk, ps=PS_)')
sub('"g170_m3.csv"', '"g252_m3.csv"'); sub('"g170_train.csv"', '"g252_train.csv"')
exec(compile(s, "g170s", "exec"))
