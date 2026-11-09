"""G295 (11/05, ledger #238 step 1): the onset temperature on M3 v3 dev onset 100 windows (hold-out days not read) by cell proper scores.
G170's flow with A17's onset (0.8 onset networks + 0.2 pcL s0-3, the physics onset adaptive a = April composition, the evidence penalty
0.25), the T+30 row just before best_set, per k with the k weight; the scored cells of the T+30 row (m["m"]). p_T = sigmoid(logit(p) / T),
T on the declared grid. Main: the mean cell log loss (private weights of m3v2.weights; unweighted also printed); secondary: Brier and the
log loss on 0.02 <= p <= 0.98. Window bootstrap 2000: the 90 % interval of T*. Identity: best_set at T = 1 reproduces G280's pc4 column.
Step 2 of the declaration: an interval containing 1.0 closes #238. usage: TFB_M3VER=v3 py -B -X utf8 g295_on_temp_logloss.py"""
import sys, pickle
s = open(r"C:/Users/user/tfb/review/g170_onset_instrument.py", encoding="utf-8").read()


def sub(a, b, n=1):
    global s
    assert s.count(a) == n, (a[:60], s.count(a))
    s = s.replace(a, b)


L1 = '            S = sets(CTX[key], PN[w], ph, a, m["t"], k, PO.adaptive_a(ev2, m["day"]))\n'
sub('"hid", "hidf", "ac"); ND = 8', '"hid", "hidf", "ac", "pc4"); ND = 8')
sub('    return {v: decode(x)[-1] for v, x in pm.items()}', '    out_ = {v: decode(x)[-1] for v, x in pm.items()}; out_["_b"] = b; return out_')
sub(L1 + '            for v in VARS:\n                sc[v] += pr * iou1(S[v], y, e)\n',
    L1 + '            for v in VARS[:-1]:\n                sc[v] += pr * iou1(S[v], y, e)\n'
    '            S4 = sets(CTX[key], 0.8 * PN[w] + 0.2 * sum(PC_[j][w] for j in range(4)) / 4, ph, a, m["t"], k, PO.adaptive_a(ev2, m["day"])); sc["pc4"] += pr * iou1(S4["base"], y, e)\n'
    '            DUMP.setdefault(w, dict(y=y, e=e, k=[]))["k"].append((pr, S4["_b"]))\n')
sub('def march(limit=None):', 'PC_ = [pickle.load(open(W + f"probs_m3v3cap__pcL_s{s_}_fold2030-12.pkl", "rb")) for s_ in range(4)]; DUMP = {}\n\n\ndef march(limit=None):')
sub("t0 = TM(); Dt = train_months(lim)", '''P8 = pd.read_csv(W + "g280_m3.csv").set_index("w"); print("identity pc4 against g280:", float(np.abs(Dm.set_index("w").pc4 - P8.pc4.reindex(Dm.w).to_numpy()).max()), flush=True)
    pickle.dump(DUMP, open(W + "g295_on_pretemp_m3.pkl", "wb")); TS = np.array([0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.5, 1.8, 2.0]); WID = list(Dm.w)
    i1 = [sum(pr * iou1(decode(b_)[-1], DUMP[w_]["y"], DUMP[w_]["e"]) for pr, b_ in DUMP[w_]["k"]) for w_ in WID]; print("identity best_set of the dumped probabilities against pc4:", float(np.abs(np.array(i1) - Dm.pc4.to_numpy()).max()), flush=True)
    Sx = np.zeros((len(WID), len(TS), 3)); Nx = np.zeros((len(WID), 2))
    for i_, w_ in enumerate(WID):
        y_, e_ = DUMP[w_]["y"], DUMP[w_]["e"]
        for pr, b_ in DUMP[w_]["k"]:
            p0 = b_[-1][e_].astype(float); yy = y_[e_].astype(float); mid = (p0 >= 0.02) & (p0 <= 0.98); z = lg(p0); Nx[i_] += pr * np.array([e_.sum(), mid.sum()])
            for j_, T_ in enumerate(TS):
                q_ = np.clip(1 / (1 + np.exp(-z / T_)), 1e-7, 1 - 1e-7); ll = -(yy * np.log(q_) + (1 - yy) * np.log(1 - q_)); Sx[i_, j_] += pr * np.array([ll.sum(), ((q_ - yy) ** 2).sum(), ll[mid].sum()])
    WT = m3v2.weights(Dm[["w", "cond", "panel", "day", "hour"]], "private")[0]
    def cv_(wt, ix):
        tot = (wt[ix][:, None, None] * Sx[ix]).sum(axis=0); n_ = (wt[ix][:, None] * Nx[ix]).sum(axis=0); return np.column_stack([tot[:, 0] / n_[0], tot[:, 1] / n_[0], tot[:, 2] / n_[1]])
    al = np.arange(len(WID))
    for nm, wt in (("weighted", WT), ("unweighted", np.ones(len(WID)))):
        c_ = cv_(wt, al); print(f"M3 A17 onset ({nm}): T* log loss {TS[c_[:, 0].argmin()]}, Brier {TS[c_[:, 1].argmin()]}, mid {TS[c_[:, 2].argmin()]} | log loss " + " ".join(f"{T_}:{v_:.5f}" for T_, v_ in zip(TS, c_[:, 0])), flush=True)
    rg = np.random.default_rng(0); tb = np.array([TS[cv_(WT, rg.integers(0, len(WID), len(WID)))[:, 0].argmin()] for _ in range(2000)]); lo, hi = np.quantile(tb, 0.05), np.quantile(tb, 0.95)
    print(f"T*_M3(A17 onset) {TS[cv_(WT, al)[:, 0].argmin()]} 90% [{lo}, {hi}] | contains 1.0: {lo <= 1.0 <= hi} -> " + ("#238 closes (step 2)" if lo <= 1.0 <= hi else "step 3: delta needed"), flush=True)
    sys.exit()
    Dt = None''')
exec(compile(s, "g170p6", "exec"))
