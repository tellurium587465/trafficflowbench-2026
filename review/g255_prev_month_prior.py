"""G255 (11/04, ledger #215, the strategy's (H) from panel 1): the previous month's onset births by link x time band as a prior for the
windows with no same-month evidence (k <= 1 only). G170's onset instrument, decode only. Births = G214's definition (a queued cell after
three free rows on the masked inputs, PanelInputs.mask_v) in the previous month: March dev windows <- February (train split), January
<- December, February <- January (the train split runs June-February without a gap). Prior(j, band) = (n_jb + 0.5) / (sum_j n_jb +
0.5 n), bands = origin hour < 9, 9-12, 12-15, >= 15; the physics onset term ph -> sigmoid(logit ph + w log(n prior)) for w = 0.25 /
0.5 (pp25 / pp50), only at k <= 1 (the kdist draw), elsewhere as the base. Readings as G170.
usage: TFB_M3VER=v3 py -B -X utf8 g255_prev_month_prior.py [time | run]"""
import sys
s = open(r"C:/Users/user/tfb/review/g170_onset_instrument.py", encoding="utf-8").read()


def sub(a, b, n=1):
    global s
    assert s.count(a) == n, (a[:60], s.count(a))
    s = s.replace(a, b)


sub('"hid", "hidf", "ac"); ND = 8', '"hid", "hidf", "ac", "pp25", "pp50"); ND = 8')
sub('def sets(c, p_, ph, a, t, k, a2=None):', 'def sets(c, p_, ph, a, t, k, a2=None, pri=None):')
sub('    return {v: decode(x)[-1] for v, x in pm.items()}\n',
    '''    for wv, nm_ in ((0.25, "pp25"), (0.5, "pp50")):
        if pri is None or k > 1:
            pm[nm_] = b
        else:
            q_ = p_.copy(); q_[5] = (1 - a) * p_[5] + a * sg(lg(ph) + wv * np.log(len(pri) * pri)); pm[nm_] = rule(q_, 0.25)
    return {v: decode(x)[-1] for v, x in pm.items()}
''')
sub('def march(limit=None):', '''from t2_model import PanelInputs as _PI
BIRTH = {}


def births_month(p, month):                                     # G214's births on the train split's masked inputs, rows of `month`
    if p not in BIRTH:
        P_ = load(p, "train"); V_ = _PI(P_).mask_v; q = (V_ <= P_.vcut[None, :]) & np.isfinite(V_); b_ = q[3:] & ~q[2:-1] & ~q[1:-2] & ~q[:-3]; r_, j_ = np.nonzero(b_); r_ = r_ + 3
        BIRTH[p] = (np.array([P_.times[x][:7] for x in r_]), np.array([(x % 288) // 12 for x in r_]), j_, P_.n)
    mo, hr, j_, n = BIRTH[p]; out = {}
    for bi, (lo, hi) in enumerate(((0, 9), (9, 12), (12, 15), (15, 24))):
        m_ = (mo == month) & (hr >= lo) & (hr < hi); cnt = np.bincount(j_[m_], minlength=n).astype(float); out[bi] = (cnt + 0.5) / (cnt.sum() + 0.5 * n)
    return out


def band(t):
    h = (t % 288) // 12; return 0 if h < 9 else 1 if h < 12 else 2 if h < 15 else 3


PREV = {"2031-03": "2031-02", "2031-01": "2030-12", "2031-02": "2031-01"}


def march(limit=None):''')
sub('            S = sets(CTX[key], PN[w], ph, a, m["t"], k, PO.adaptive_a(ev2, m["day"]))',
    '            S = sets(CTX[key], PN[w], ph, a, m["t"], k, PO.adaptive_a(ev2, m["day"]), pri=births_month(m["panel"], PREV["2031-03"])[band(m["t"])])')
sub('S = sets(c, p_, ph, a, t, kk)', 'S = sets(c, p_, ph, a, t, kk, pri=births_month(p, PREV[mo])[band(t)])')
sub('"g170_m3.csv"', '"g255_m3.csv"'); sub('"g170_train.csv"', '"g255_train.csv"')
exec(compile(s, "g170p", "exec"))
