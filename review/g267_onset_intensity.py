"""G267 (11/04, ledger #226, declared by the strategy before the results, auditor L1): the onset evidence rule from a box to an
intensity. G170's onset instrument (final A's onset flow, decode only). The current rule: links within +-2 of a queue seen on the
month's earlier days in rows T-30 .. T+55 of the clock keep their T+30 probability, the others x0.25. The arms: lambda_j = the visible
queue on the earlier days (the same days and hidden rows as the rule, at most 3 back), weighted by a Gaussian in time centred on the
T+30 clock (sigma 60 or 180 min) and in space (sigma 1 link); eps = min(1, lambda_j / lambda_ref), lambda_ref = the p50 / p60 / p75 of
the corridor's positive lambda (eps = 0 when none); the T+30 row x (0.25 + 0.75 eps) when earlier days exist (as the rule). Six arms
I{p}_{s}. Readings as G170: M3 dev (hold-out days not read) and January / February L / F / M. The odd / even day read is G268.
usage: TFB_M3VER=v3 py -B -X utf8 g267_onset_intensity.py [time | run]"""
import sys
s = open(r"C:/Users/user/tfb/review/g170_onset_instrument.py", encoding="utf-8").read()
ARMS = [f"I{p}_{sg}" for p in (50, 60, 75) for sg in (60, 180)]


def sub(a, b, n=1):
    global s
    assert s.count(a) == n, (a[:60], s.count(a))
    s = s.replace(a, b)


sub('"hid", "hidf", "ac"); ND = 8', '"hid", "hidf", "ac", ' + ", ".join(f'"{a}"' for a in ARMS) + '); ND = 8')
sub('    return {v: decode(x)[-1] for v, x in pm.items()}\n',
    '''    for sg_ in (60, 180):
        L_, nd_ = lam(c, t, k, sg_ // 5)
        pos_ = L_[L_ > 0]
        for p_q in (50, 60, 75):
            if not nd_:
                pm[f"I{p_q}_{sg_}"] = pa
                continue
            ref_ = np.percentile(pos_, p_q) if pos_.size else 0.0; eps_ = np.minimum(1.0, L_ / ref_) if ref_ > 0 else np.zeros_like(L_)
            x_ = pa.copy(); x_[-1] = x_[-1] * (0.25 + 0.75 * eps_); pm[f"I{p_q}_{sg_}"] = x_
    return {v: decode(x)[-1] for v, x in pm.items()}


KS_ = np.exp(-0.5 * np.arange(-3, 4) ** 2)


def lam(c, t, k, sr):                                           # the intensity of the visible queue on the month's earlier days
    hid = sim_blank(c.key, t // 288, min(3, k), t); L = np.zeros(c.n); nd = 0
    for back in range(1, min(3, k) + 1):
        dd = t - back * 288
        if dd - 6 < 0:                                          # as the rule (seen with lo = -6)
            break
        ctr = dd + 6; rr = np.array([r for r in range(max(0, ctr - 3 * sr), ctr + 3 * sr + 1) if r not in hid and r < t], int)
        if rr.size:
            w = np.exp(-0.5 * ((rr - ctr) / sr) ** 2); L = L + (w[:, None] * c.q[rr]).sum(axis=0)
        nd += 1
    return np.convolve(L, KS_, mode="same"), nd
''')
sub('"g170_m3.csv"', '"g267_m3.csv"'); sub('"g170_train.csv"', '"g267_train.csv"')
exec(compile(s, "g170i", "exec"))
