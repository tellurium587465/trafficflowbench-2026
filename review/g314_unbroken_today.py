"""G314 (11/05, ledger #248): "not broken yet today". For each onset window and link j, the exposure H_j accumulated on the window's own day
before T: H_j = sum over today's visible rows r < T where j flows freely (v > vcut) of lam[j, slot(r)] * q[r, j] / cap_j, with lam the hazard of
a link-level onset per unit of free-flow exposure from the training months (onset at j: queued at r after >= 12 visible free rows; exposure:
sum of q / cap over the visible free rows; site x 15-minute slot, plus one pseudo-slot of the site's mean; January / February windows use the
table without their own month). Gamma-Poisson: the month's activity of a site varies with shape k, estimated by moments from the per-site
monthly onset counts of the training months (pooled: k = sum mu^2 / sum (var - mu)); no event seen today -> the posterior multiplier
1 / (1 + H_j / k) on p_j at T+30 (after the physics mix, before the evidence rule). G170's flow: M3 v3 dev onset 100 windows with A17's onset
(pc4) and the 317 January / February onset windows (OOF flow, L). Line (#247-#252): both points > 0, merged z >= +1.
usage: TFB_M3VER=v3 py -B -X utf8 g314_unbroken_today.py"""
import sys, pickle
s = open(r"C:/Users/user/tfb/review/g170_onset_instrument.py", encoding="utf-8").read()


def sub(a, b, n=1):
    global s
    assert s.count(a) == n, (a[:60], s.count(a))
    s = s.replace(a, b)


PRE = '''
HZ = {}


def hazard_tables():
    """per panel: lam[(month_excluded)] (n_links, 96) and the per-site monthly onset counts; k from all training months"""
    out, cnt = {}, []
    for p in T2_PANELS:
        P = load(p, "train"); V, Q = P.V, P.Q; fin = np.isfinite(V); qd = fin & (V <= P.vcut[None, :]); fr = fin & ~qd
        qc = np.where(fr & np.isfinite(Q), Q / P.cap[None, :], 0.0); mon = np.array([str(x)[:7] for x in P.times]); slot = (np.arange(P.T) % 288) // 3
        free12 = np.zeros_like(fr)
        run = np.zeros(P.n, int)
        for r in range(P.T):
            free12[r] = run >= 12; run = np.where(fr[r], run + 1, np.where(fin[r], 0, run))
        ons = qd & free12
        ms = sorted(set(mon)); B = {m_: np.zeros((P.n, 96)) for m_ in ms}; E = {m_: np.zeros((P.n, 96)) for m_ in ms}
        for m_ in ms:
            rr = mon == m_
            for sl in range(96):
                ix = rr & (slot == sl); B[m_][:, sl] = ons[ix].sum(axis=0); E[m_][:, sl] = qc[ix].sum(axis=0)
        cnt.append(np.stack([B[m_].sum(axis=1) for m_ in ms], axis=1))
        lam = {}
        for ex in [None] + ms:
            b = sum(B[m_] for m_ in ms if m_ != ex); e = sum(E[m_] for m_ in ms if m_ != ex)
            bj, ej = b.sum(axis=1, keepdims=True) / 96, e.sum(axis=1, keepdims=True) / 96; lam[ex] = (b + bj) / np.maximum(e + ej, 1e-9)
        out[p] = lam; load.cache_clear()
    C = np.vstack(cnt); mu = C.mean(axis=1); va = C.var(axis=1, ddof=1); keep = mu > 0
    k = float((mu[keep] ** 2).sum() / max((va[keep] - mu[keep]).sum(), 1e-9))
    print(f"hazard tables: {len(out)} panels, sites {len(C)}; per-site monthly onsets mean {mu.mean():.2f}, var {va.mean():.2f} -> Gamma shape k {k:.3f}", flush=True)
    return out, k


LAM, KSH = hazard_tables()


def unbroken(c, t, k, lam):
    """multiplier 1 / (1 + H / k) per link from today's visible free rows before T"""
    d0 = t - t % 288; hid = sim_blank(c.key, t // 288, min(3, k), t); rr = [r for r in range(d0, t) if r not in hid]
    if not rr:
        return np.ones(c.n)
    V = c.ns.V[rr]; Q = QC[c.key][rr]; fr = np.isfinite(V) & (V > VC[c.key][None, :]) & np.isfinite(Q)
    sl = (np.array(rr) % 288) // 3; H = (np.where(fr, Q, 0.0) * lam[:, sl].T).sum(axis=0); return 1.0 / (1.0 + H / KSH)

'''
sub('def sets(c, p_, ph, a, t, k, a2=None):', PRE + '\ndef sets(c, p_, ph, a, t, k, a2=None):')
sub('"hid", "hidf", "ac"); ND = 8', '"hid", "hidf", "ac", "ub"); ND = 8')
sub('    return {v: decode(x)[-1] for v, x in pm.items()}',
    '    u_ = unbroken(c, t, k, LAM_CUR[0]); x_ = pa.copy(); x_[-1] = x_[-1] * u_; pm["ub"] = rule(x_, 0.25); UBS.append(float(u_.min()))\n'
    '    return {v: decode(x)[-1] for v, x in pm.items()}')
sub('            S = sets(CTX[key], PN[w], ph, a, m["t"], k, PO.adaptive_a(ev2, m["day"]))\n',
    '            LAM_CUR[0] = LAM[m["panel"]][None]; S = sets(CTX[key], 0.8 * PN[w] + 0.2 * sum(PC_[j][w] for j in range(4)) / 4, ph, a, m["t"], k, PO.adaptive_a(ev2, m["day"]))\n')
sub('                    S = sets(c, p_, ph, a, t, kk)\n', '                    LAM_CUR[0] = LAM[p][mo]; S = sets(c, p_, ph, a, t, kk)\n')
sub('            CTX[key] = Ctx(key, P, P.V)\n', '            CTX[key] = Ctx(key, P, P.V); QC[key] = P.Q / P.cap[None, :]; VC[key] = P.vcut\n')
sub('CTX[key] = Ctx(key, P, np.where(P.blank, np.nan, P.V), P.V); c = CTX[key];', 'CTX[key] = Ctx(key, P, np.where(P.blank, np.nan, P.V), P.V); c = CTX[key]; QC[key] = np.where(P.blank, np.nan, P.Q) / P.cap[None, :]; VC[key] = P.vcut;')
sub('def march(limit=None):', 'PC_ = [pickle.load(open(W + f"probs_m3v3cap__pcL_s{s_}_fold2030-12.pkl", "rb")) for s_ in range(4)]; LAM_CUR = [None]; QC = {}; VC = {}; UBS = []\n\n\ndef march(limit=None):')
sub("t0 = TM(); Dm = march(lim)", "t0 = TM(); Dm = march(lim); P8 = pd.read_csv(W + 'g280_m3.csv').set_index('w'); print('identity base (A17 onset) against g280 pc4:', float(np.abs(Dm.set_index('w').base - P8.pc4.reindex(Dm.w).to_numpy()).max()), flush=True)")
sub("o = pd.read_csv(W + \"g108_rectangles.csv\").set_index(\"w\").A.reindex(Dm.w).to_numpy()", "o = Dm.base.to_numpy()")
sub("t0 = TM(); Dt = train_months(lim)", '''t0 = TM(); Dt = train_months(lim); Dm.to_csv(W + "g314_m3.csv", index=False); Dt.to_csv(W + "g314_jf.csv", index=False); print("multiplier min over links: median", round(float(np.median(UBS)), 3), flush=True)
    d1 = (Dm.ub - Dm.base).to_numpy(); r1 = m3v2.summary(Dm.assign(cond="queue_onset"), "base", "ub", "private"); a1, s1 = r1["t2_weighted"], r1["se"]
    cl = pd.factorize(Dt.month + "_" + Dt.panel + "_" + Dt.day.astype(str))[0]; Rm = np.random.default_rng(0).multinomial(cl.max() + 1, np.full(cl.max() + 1, 1 / (cl.max() + 1)), size=2000)
    d2 = (Dt.L_ub - Dt.L_base).to_numpy(); a2 = d2.mean(); s2 = float(((Rm @ np.bincount(cl, d2)) / (Rm @ np.bincount(cl))).std())
    w1, w2 = 1 / max(s1, 1e-9) ** 2, 1 / max(s2, 1e-9) ** 2; mg = (w1 * a1 + w2 * a2) / (w1 + w2); sm = (w1 + w2) ** -0.5
    print(f"M3 dev onset (A17): ub - base {a1:+.5f} (SE {s1:.5f}, z {a1 / max(s1, 1e-12):+.2f}, raw {d1.mean():+.5f}, windows changed {int((np.abs(d1) > 1e-12).sum())}) | Jan / Feb onset (L): {a2:+.5f} (SE {s2:.5f}, z {a2 / max(s2, 1e-12):+.2f}, changed {int((np.abs(d2) > 1e-12).sum())}) | merged {mg:+.5f} (z {mg / sm:+.2f}) | line: {a1 > 0 and a2 > 0 and mg / sm >= 1}", flush=True)
    sys.exit()
    Dt = None''')
exec(compile(s, "g170p13", "exec"))
