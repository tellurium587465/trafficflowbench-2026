"""G317 (11/05, ledger #253): how the physics ranker's probability is mixed into onset (the ranker unchanged). G170's flow with A17's onset on
M3 v3 dev onset 100 windows and the OOF flow on the 317 January / February onset windows (L labels). The window's adaptive a (G170: the month's
earlier births, capped as there) is scaled by f with the cap scaled alike (cap 0.5 -> 0.5 f):
  lin1.0 (today), lin1.5, lin2.0 : p5 = (1 - a f) p_NN + a f p_rank
  log1.0, log1.5                  : logit-linear (normalised geometric) mix, logit p5 = (1 - a f) logit p_NN + a f logit p_rank
then the evidence penalty 0.25 and best_set as base. Selection on M3 dev odd days + January (the arm with the largest inverse-variance merged
gain over lin1.0, lin1.0 itself allowed); the chosen arm read on M3 dev even days + February; line (#247-#252): both > 0, merged z >= +1.
usage: TFB_M3VER=v3 py -B -X utf8 g317_rank_mix.py"""
import sys, pickle
s = open(r"C:/Users/user/tfb/review/g170_onset_instrument.py", encoding="utf-8").read()
ARMS = ("lin1.5", "lin2.0", "log1.0", "log1.5")


def sub(a, b, n=1):
    global s
    assert s.count(a) == n, (a[:60], s.count(a))
    s = s.replace(a, b)


sub('"hid", "hidf", "ac"); ND = 8', '"hid", "hidf", "ac", ' + ", ".join(f'"{v}"' for v in ARMS) + '); ND = 8')
sub('    return {v: decode(x)[-1] for v, x in pm.items()}',
    '    for f_ in (1.5, 2.0):\n'
    '        af = min(a * f_, 0.5 * f_); q_ = p_.copy(); q_[5] = (1 - af) * p_[5] + af * ph; pm[f"lin{f_}"] = rule(q_, 0.25)\n'
    '    for f_ in (1.0, 1.5):\n'
    '        af = min(a * f_, 0.5 * f_); q_ = p_.copy(); q_[5] = sg((1 - af) * lg(p_[5]) + af * lg(ph)); pm[f"log{f_}"] = rule(q_, 0.25)\n'
    '    return {v: decode(x)[-1] for v, x in pm.items()}')
sub('            S = sets(CTX[key], PN[w], ph, a, m["t"], k, PO.adaptive_a(ev2, m["day"]))\n',
    '            S = sets(CTX[key], 0.8 * PN[w] + 0.2 * sum(PC_[j][w] for j in range(4)) / 4, ph, a, m["t"], k, PO.adaptive_a(ev2, m["day"]))\n')
sub('def march(limit=None):', 'PC_ = [pickle.load(open(W + f"probs_m3v3cap__pcL_s{s_}_fold2030-12.pkl", "rb")) for s_ in range(4)]\n\n\ndef march(limit=None):')
sub("t0 = TM(); Dm = march(lim)", "t0 = TM(); Dm = march(lim); P8 = pd.read_csv(W + 'g280_m3.csv').set_index('w'); print('identity base (A17 onset) against g280 pc4:', float(np.abs(Dm.set_index('w').base - P8.pc4.reindex(Dm.w).to_numpy()).max()), flush=True)")
sub("o = pd.read_csv(W + \"g108_rectangles.csv\").set_index(\"w\").A.reindex(Dm.w).to_numpy()", "o = Dm.base.to_numpy()")
sub("t0 = TM(); Dt = train_months(lim)", '''t0 = TM(); Dt = train_months(lim); Dm.to_csv(W + "g317_m3.csv", index=False); Dt.to_csv(W + "g317_jf.csv", index=False)
    rg = np.random.default_rng(0)
    def m3(D_, v_):
        r_ = m3v2.summary(D_, "base", v_, "private"); return r_["t2_weighted"], r_["se"]
    def jf(D_, v_):
        cl = pd.factorize(D_.month + "_" + D_.panel + "_" + D_.day.astype(str))[0]; Rm = rg.multinomial(cl.max() + 1, np.full(cl.max() + 1, 1 / (cl.max() + 1)), size=2000)
        d_ = (D_["L_" + v_] - D_.L_base).to_numpy(); return d_.mean(), float(((Rm @ np.bincount(cl, d_)) / (Rm @ np.bincount(cl))).std())
    def mg(x, y):
        w1, w2 = 1 / max(x[1], 1e-9) ** 2, 1 / max(y[1], 1e-9) ** 2; m_ = (w1 * x[0] + w2 * y[0]) / (w1 + w2); return m_, (w1 + w2) ** -0.5
    odd_m = Dm[(Dm.day + 1) % 2 == 1].reset_index(drop=True); evn_m = Dm[(Dm.day + 1) % 2 == 0].reset_index(drop=True); jan = Dt[Dt.month == "2031-01"].reset_index(drop=True); feb = Dt[Dt.month == "2031-02"].reset_index(drop=True)
    sel = {}
    for v_ in ''' + str(ARMS) + ''':
        x, y = m3(odd_m, v_), jf(jan, v_); sel[v_] = mg(x, y); print(f"select {v_}: M3 odd {x[0]:+.5f} ({x[1]:.5f}) | Jan {y[0]:+.5f} ({y[1]:.5f}) | merged {sel[v_][0]:+.5f} (z {sel[v_][0] / sel[v_][1]:+.2f})", flush=True)
    best = max(sel, key=lambda v_: sel[v_][0])
    if sel[best][0] <= 0:
        print("selection: lin1.0 (today) stays (no arm above it)", flush=True); sys.exit()
    x, y = m3(evn_m, best), jf(feb, best); m_, s_ = mg(x, y)
    print(f"read {best}: M3 even {x[0]:+.5f} (SE {x[1]:.5f}) | Feb {y[0]:+.5f} (SE {y[1]:.5f}) | merged {m_:+.5f} (z {m_ / s_:+.2f}) | line: {x[0] > 0 and y[0] > 0 and m_ / s_ >= 1}", flush=True)
    sys.exit()
    Dt = None''')
exec(compile(s, "g170p15", "exec"))
