"""G303 (11/05, ledger #242 stage 0 (a), (b)): the onset evidence penalty (1.0 - 0.25) by day type. (a) M3 v3 dev onset 100 windows, G170's
flow with A17's onset (0.8 onset networks + 0.2 pcL s0-3, physics adaptive a): B1.0 (no penalty) against base (0.25), k expectation. (b) the
317 January / February selector onset windows, G170's saved columns L_B1.0 - L_base (g170_train.csv, the onset networks' OOF flow). Strata:
the window's day type (weekday / weekend = Sat, Sun) and "mismatch" = every evidence day (the up to 3 earlier days of the month) of the other
type. Per stratum: windows, mean, window bootstrap SE (2000). Line (#242): the mismatch stratum (else weekend) > 0 on both, the inverse-variance
merge z >= +1, and the M3 weekday stratum <= 0. usage: TFB_M3VER=v3 py -B -X utf8 g303_weekday_penalty.py"""
import sys, pickle
s = open(r"C:/Users/user/tfb/review/g170_onset_instrument.py", encoding="utf-8").read()


def sub(a, b, n=1):
    global s
    assert s.count(a) == n, (a[:60], s.count(a))
    s = s.replace(a, b)


L1 = '            S = sets(CTX[key], PN[w], ph, a, m["t"], k, PO.adaptive_a(ev2, m["day"]))\n'
sub('"hid", "hidf", "ac"); ND = 8', '"hid", "hidf", "ac", "pc4", "pc4B1"); ND = 8')
sub(L1 + '            for v in VARS:\n                sc[v] += pr * iou1(S[v], y, e)\n',
    L1 + '            for v in VARS[:-2]:\n                sc[v] += pr * iou1(S[v], y, e)\n'
    '            S4 = sets(CTX[key], 0.8 * PN[w] + 0.2 * sum(PC_[j][w] for j in range(4)) / 4, ph, a, m["t"], k, PO.adaptive_a(ev2, m["day"])); sc["pc4"] += pr * iou1(S4["base"], y, e); sc["pc4B1"] += pr * iou1(S4["B1.0"], y, e)\n')
sub('def march(limit=None):', 'PC_ = [pickle.load(open(W + f"probs_m3v3cap__pcL_s{s_}_fold2030-12.pkl", "rb")) for s_ in range(4)]\n\n\ndef march(limit=None):')
sub("t0 = TM(); Dt = train_months(lim)", '''P8 = pd.read_csv(W + "g280_m3.csv").set_index("w"); print("identity pc4 against g280:", float(np.abs(Dm.set_index("w").pc4 - P8.pc4.reindex(Dm.w).to_numpy()).max()), flush=True)
    wk = lambda dt: pd.Timestamp(dt).dayofweek >= 5
    def strata(month, days):
        typ = np.array([wk(f"{month}-{d + 1:02d}") for d in days]); mis = np.array([d > 0 and all(wk(f"{month}-{d - b + 1:02d}") != t_ for b in (1, 2, 3) if d - b >= 0) for d, t_ in zip(days, typ)]); return typ, mis
    def se(d_):
        rg_ = np.random.default_rng(0); return float(np.std([d_[rg_.integers(0, len(d_), len(d_))].mean() for _ in range(2000)])) if len(d_) > 1 else float("nan")
    R = {}
    dM = (Dm.pc4B1 - Dm.pc4).to_numpy(); tM, mM = strata("2031-03", Dm.day.to_numpy())
    Dj = pd.read_csv(W + "g170_train.csv"); dJ = (Dj["L_B1.0"] - Dj.L_base).to_numpy(); tJ = np.zeros(len(Dj), bool); mJ = np.zeros(len(Dj), bool)
    for mo in ("2031-01", "2031-02"):
        ix = (Dj.month == mo).to_numpy(); a_, b_ = strata(mo, Dj.day.to_numpy()[ix]); tJ[ix] = a_; mJ[ix] = b_
    for nm, d_, t_, m_ in (("M3 dev onset (A17)", dM, tM, mM), ("Jan / Feb onset (OOF flow, L)", dJ, tJ, mJ)):
        out = []
        for lab, sel in (("all", np.ones(len(d_), bool)), ("weekday", ~t_), ("weekend", t_), ("mismatch", m_), ("weekend, no mismatch", t_ & ~m_), ("weekday, mismatch", ~t_ & m_)):
            v = d_[sel]; R[(nm, lab)] = (v.mean() if len(v) else float("nan"), se(v), len(v)); out.append(f"{lab} {len(v)}: {R[(nm, lab)][0]:+.4f} (SE {R[(nm, lab)][1]:.4f})")
        print(f"{nm}, penalty 1.0 - 0.25 | " + " | ".join(out), flush=True)
    key = "mismatch" if R[("M3 dev onset (A17)", "mismatch")][2] and R[("Jan / Feb onset (OOF flow, L)", "mismatch")][2] else "weekend"
    (a1, s1, n1), (a2, s2, n2) = R[("M3 dev onset (A17)", key)], R[("Jan / Feb onset (OOF flow, L)", key)]; w1, w2 = 1 / max(s1, 1e-9) ** 2, 1 / max(s2, 1e-9) ** 2; mg = (w1 * a1 + w2 * a2) / (w1 + w2); sm = (w1 + w2) ** -0.5
    wd = R[("M3 dev onset (A17)", "weekday")][0]
    print(f"line on the {key} stratum: M3 {a1:+.4f} > 0 {a1 > 0}, JF {a2:+.4f} > 0 {a2 > 0}, merged {mg:+.4f} (SE {sm:.4f}, z {mg / sm:+.2f}) >= +1 {mg / sm >= 1}, M3 weekday {wd:+.4f} <= 0 {wd <= 0} -> {a1 > 0 and a2 > 0 and mg / sm >= 1 and wd <= 0}", flush=True)
    sys.exit()
    Dt = None''')
exec(compile(s, "g170p9", "exec"))
