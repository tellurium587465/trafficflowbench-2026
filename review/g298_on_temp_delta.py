"""G298 (11/05, ledger #238 step 3): the instrument shift of the onset temperature. G170's onset flow (the physics onset with the adaptive a,
the evidence penalty 0.25, best_set; on January / February the leave-one-month-out physics model as G170's train_months) with the onset
networks replaced by capB0 s0-3: on M3 v3 dev onset 100 windows (probs_m3v3__capB0, the masked inputs of M3) and on the 317 January /
February selector onset windows (probs_jfonmh0__capB0, G296's unmasked-history inputs, CPU inference). The T+30 row just before best_set per
k with the k weight; the scored cells: M3 m["m"], January / February the label eligibility of row 5 (G170's L). Also A17's onset on M3 (as
G295) so the M3 windows are resampled jointly. Cell log loss on the #238 grid, T* = argmin, window bootstrap 2000 (M3 jointly, January /
February independently): T*_M3(A17), delta = T*_JF(capB0) - T*_M3(capB0), T*_M3(A17) + delta; the line of #238 (both ends exclude 1.0 on
the same side -> the end nearer 1.0). Identity: A17's T = 1 decode against G280's pc4. usage: TFB_M3VER=v3 py -B -X utf8 g298_on_temp_delta.py"""
import sys, pickle
s = open(r"C:/Users/user/tfb/review/g170_onset_instrument.py", encoding="utf-8").read()


def sub(a, b, n=1):
    global s
    assert s.count(a) == n, (a[:60], s.count(a))
    s = s.replace(a, b)


L1 = '            S = sets(CTX[key], PN[w], ph, a, m["t"], k, PO.adaptive_a(ev2, m["day"]))\n'
sub('"hid", "hidf", "ac"); ND = 8', '"hid", "hidf", "ac", "pc4"); ND = 8')
sub('    return {v: decode(x)[-1] for v, x in pm.items()}', '    out_ = {v: decode(x)[-1] for v, x in pm.items()}; out_["_b"] = b; out_["pc4"] = out_["base"]; return out_')
sub(L1 + '            for v in VARS:\n                sc[v] += pr * iou1(S[v], y, e)\n',
    L1 + '            for v in VARS[:-1]:\n                sc[v] += pr * iou1(S[v], y, e)\n'
    '            S4 = sets(CTX[key], 0.8 * PN[w] + 0.2 * sum(PC_[j][w] for j in range(4)) / 4, ph, a, m["t"], k, PO.adaptive_a(ev2, m["day"])); sc["pc4"] += pr * iou1(S4["base"], y, e)\n'
    '            SB = sets(CTX[key], sum(CB_[j][w] for j in range(4)) / 4, ph, a, m["t"], k, PO.adaptive_a(ev2, m["day"]))\n'
    '            DM_.setdefault(w, dict(y=y, e=e, A=[], C=[])); DM_[w]["A"].append((pr, S4["_b"][-1])); DM_[w]["C"].append((pr, SB["_b"][-1]))\n')
sub('def march(limit=None):', 'PC_ = [pickle.load(open(W + f"probs_m3v3cap__pcL_s{s_}_fold2030-12.pkl", "rb")) for s_ in range(4)]; CB_ = [pickle.load(open(W + f"probs_m3v3__capB0_s{s_}_fold2030-12.pkl", "rb")) for s_ in range(4)]\n'
    'CJ_ = [pickle.load(open(W + f"probs_jfonmh0__capB0_s{s_}_fold2030-12.pkl", "rb")) for s_ in range(4)]; DM_ = {}; DJ_ = {}\n\n\ndef march(limit=None):')
sub('p_ = (OOF[0][k] + OOF[1][k]) / 2;', 'p_ = sum(CJ_[j][str(k)] for j in range(4)) / 4;')
sub('                    S = sets(c, p_, ph, a, t, kk)\n', '                    S = sets(c, p_, ph, a, t, kk); DJ_.setdefault(int(k), dict(y=yL, e=eL, C=[]))["C"].append((pr, S["_b"][-1]))\n')
sub("t0 = TM(); Dt = train_months(lim)", '''P8 = pd.read_csv(W + "g280_m3.csv").set_index("w"); print("identity pc4 against g280:", float(np.abs(Dm.set_index("w").pc4 - P8.pc4.reindex(Dm.w).to_numpy()).max()), flush=True)
    t0 = TM(); Dt = train_months(lim); print(f"Jan / Feb onset windows {len(DJ_)} ({TM() - t0:.0f}s)", flush=True)
    pickle.dump(dict(M=DM_, J=DJ_), open(W + "g298_on_pretemp.pkl", "wb")); TS = np.array([0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.5, 1.8, 2.0])
    def tab(E):
        S_ = np.zeros((len(E), len(TS))); N_ = np.zeros(len(E))
        for i_, (y_, e_, ks_) in enumerate(E):
            yy = y_[e_].astype(float)
            for pr, b_ in ks_:
                z = lg(b_[e_].astype(float)); N_[i_] += pr * e_.sum()
                for j_, T_ in enumerate(TS):
                    q_ = np.clip(1 / (1 + np.exp(-z / T_)), 1e-7, 1 - 1e-7); S_[i_, j_] += pr * float(-(yy * np.log(q_) + (1 - yy) * np.log(1 - q_)).sum())
        return S_, N_
    WM = list(Dm.w); SA, NA = tab([(DM_[w_]["y"], DM_[w_]["e"], DM_[w_]["A"]) for w_ in WM]); SC, NC = tab([(DM_[w_]["y"], DM_[w_]["e"], DM_[w_]["C"]) for w_ in WM]); SJ, NJ = tab([(v_["y"], v_["e"], v_["C"]) for v_ in DJ_.values()])
    WT = m3v2.weights(Dm[["w", "cond", "panel", "day", "hour"]], "private")[0]; oJ = np.ones(len(SJ))
    ts = lambda S_, N_, wt, ix: TS[((wt[ix][:, None] * S_[ix]).sum(axis=0) / (wt[ix] * N_[ix]).sum()).argmin()]
    aM, aJ = np.arange(len(SA)), np.arange(len(SJ))
    for nm_, S_, N_, wt in (("M3 A17", SA, NA, WT), ("M3 capB0", SC, NC, WT), ("JF capB0", SJ, NJ, oJ)):
        c_ = (wt[:, None] * S_).sum(axis=0) / (wt * N_).sum(); print(f"{nm_}: T* {TS[c_.argmin()]} | log loss " + " ".join(f"{T_}:{v_:.5f}" for T_, v_ in zip(TS, c_)), flush=True)
    rg = np.random.default_rng(0); tA, tC, tJ = [], [], []
    for _ in range(2000):
        ix = rg.integers(0, len(SA), len(SA)); jx = rg.integers(0, len(SJ), len(SJ)); tA.append(ts(SA, NA, WT, ix)); tC.append(ts(SC, NC, WT, ix)); tJ.append(ts(SJ, NJ, oJ, jx))
    tA, tC, tJ = map(np.array, (tA, tC, tJ)); dl = tJ - tC; tAd = tA + dl; q = lambda v_: (float(np.quantile(v_, 0.05)), float(np.quantile(v_, 0.95)))
    TA0 = ts(SA, NA, WT, aM); DL0 = ts(SJ, NJ, oJ, aJ) - ts(SC, NC, WT, aM); iA, iD, iAd = q(tA), q(dl), q(tAd); side = lambda i_: "above" if i_[0] > 1.0 else ("below" if i_[1] < 1.0 else "contains")
    print(f"T*_M3(A17) {TA0} 90% [{iA[0]}, {iA[1]}] ({side(iA)} 1.0) | delta {DL0:+.2f} 90% [{iD[0]:+.2f}, {iD[1]:+.2f}] | T*_M3(A17) + delta {TA0 + DL0:.2f} 90% [{iAd[0]:.2f}, {iAd[1]:.2f}] ({side(iAd)} 1.0)", flush=True)
    ok = side(iA) == side(iAd) and side(iA) != "contains"; cand = min((TA0, TA0 + DL0), key=lambda v_: abs(v_ - 1.0)) if ok else 1.0
    print(f"#238 line: both exclude 1.0 on the same side: {ok} -> candidate onset T {round(cand / 0.05) * 0.05:.2f}" + ("" if ok else " (1.0 stays, #238 closes)"), flush=True)
    sys.exit()
    Dt = None''')
exec(compile(s, "g170p7", "exec"))
