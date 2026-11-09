"""G347 (11/05, ledger #275 data): per-window dumps for the error-analysis figures. For each window, the k draw of the largest weight:
the probabilities the decode sees (6 x n) and the decoded set (6 x n), the truth / scored cells (6 x n), the panel, T, the day, and for onset
whether the evidence penalty changes the decoded set (as G343). The k-expected IoU per window goes to a CSV.
  on_m3jf : G170's flow; M3 v3 dev onset (A17 = 0.8 q24f + 0.2 pcL 4 seeds) and January / February onset with capB0 s0-3 (as G298)
  on_jun80: G170's flow on the 80 official June windows (G342b's augmented evaluator), cvf fold 2030-06 (as G321)
  og_m3jf : G328's ongoing flows; M3 (A17's mix, G284) and January / February (capB0)
  og_jun80: G328's June ongoing flow on the 80 official June windows (cvf)
-> work/t2/g347_<mode>.pkl (+ g347_<mode>_*.csv). usage: TFB_M3VER=v3 py -B -X utf8 g347_dump.py <mode>"""
import os, sys, pickle
MODE = sys.argv[1]; R0 = "C:/Users/user/tfb/"; W = R0 + "work/t2/"
if MODE.endswith("jun80"):
    g = open(R0 + "review/g342b_jun80_run.py", encoding="utf-8").read(); g = g[:g.index("src = open(")]; sys.argv = ["g342b", "none"]; exec(compile(g, "g342b_head", "exec"))
sys.argv = sys.argv[:1]


def subber(s):
    def sub(a, b, n=1):
        assert s[0].count(a) == n, (a[:70], s[0].count(a))
        s[0] = s[0].replace(a, b)
    return sub


if MODE.startswith("on"):
    s = [open(R0 + "review/g170_onset_instrument.py", encoding="utf-8").read()]; sub = subber(s)
    sub('    return {v: decode(x)[-1] for v, x in pm.items()}', '    out_ = {v: decode(x)[-1] for v, x in pm.items()}; out_["_b"] = b.astype(np.float32); out_["_set"] = decode(b); out_["_chg"] = float((out_["base"] != out_["B1.0"]).any()); return out_')
    sub('            for v in VARS:\n                sc[v] += pr * iou1(S[v], y, e)\n',
        '            if pr > DON_.get(w, (-1,))[0]:\n                DON_[w] = (pr, k, S["_b"], S["_set"], S["_chg"], m["y"].T.astype(bool), m["m"].T.astype(bool), m["panel"], int(m["t"]), int(m["day"]), "validation")\n'
        '            for v in VARS:\n                sc[v] += pr * iou1(S[v], y, e)\n')
    sub('                    S = sets(c, p_, ph, a, t, kk)\n',
        '                    S = sets(c, p_, ph, a, t, kk)\n                    if pr > DON_.get(int(k), (-1,))[0]:\n                        DON_[int(k)] = (pr, kk, S["_b"], S["_set"], S["_chg"], lab.astype(bool), el.astype(bool), p, int(t), int(dom), "train")\n')
    sub('            S = sets(CTX[key], PN[w], ph, a, m["t"], k, PO.adaptive_a(ev2, m["day"]))\n',
        '            S = sets(CTX[key], 0.8 * PN[w] + 0.2 * sum(PC_[j][w] for j in range(4)) / 4, ph, a, m["t"], k, PO.adaptive_a(ev2, m["day"]))\n')
    sub('def march(limit=None):', 'PC_ = [pickle.load(open(W + f"probs_m3v3cap__pcL_s{s_}_fold2030-12.pkl", "rb")) for s_ in range(4)]; DON_ = {}\n\n\ndef march(limit=None):')
    if MODE == "on_jun80":
        sub('p_ = (OOF[0][k] + OOF[1][k]) / 2;', 'p_ = (CV_[0][str(k)] + CV_[1][str(k)]) / 2;')
        sub('    for mo in MONTHS:\n        mdl = phys_lomo', '    for mo in ("2030-06",):\n        mdl = phys_lomo')
        sub('def march(limit=None):', 'CV_ = [pickle.load(open(W + f"probs_jun80__cvf_attnglob_s{s_}_fold2030-06.pkl", "rb")) for s_ in (0, 1)]\n\n\ndef march(limit=None):')
        sub("t0 = TM(); Dm = march(lim)", "t0 = TM(); Dm = None")
        sub('    print(f"M3 dev {len(Dm)} windows {TM() - t0:.0f}s | identity against g108 A: {np.abs(o - Dm.base.to_numpy()).max():.2e}", flush=True); ', '    ')
        sub("o = pd.read_csv(W + \"g108_rectangles.csv\").set_index(\"w\").A.reindex(Dm.w).to_numpy()", "o = None")
        CHK = 'pd.read_csv(W + "g342_jun80_onset.csv")'
    else:
        sub('p_ = (OOF[0][k] + OOF[1][k]) / 2;', 'p_ = sum(CJ_[j][str(k)] for j in range(4)) / 4;')
        sub('def march(limit=None):', 'CJ_ = [pickle.load(open(W + f"probs_jfonmh0__capB0_s{s_}_fold2030-12.pkl", "rb")) for s_ in range(4)]\n\n\ndef march(limit=None):')
        sub("o = pd.read_csv(W + \"g108_rectangles.csv\").set_index(\"w\").A.reindex(Dm.w).to_numpy()", "o = Dm.base.to_numpy()")
        CHK = None
    sub("t0 = TM(); Dt = train_months(lim)", f'''t0 = TM(); Dt = train_months(lim)
    pickle.dump(DON_, open(W + "g347_{MODE}.pkl", "wb")); Dt[["w", "panel", "day", "L_base", "L_B1.0"]].to_csv(W + "g347_{MODE}_t.csv", index=False)
    if Dm is not None:
        Dm[["w", "panel", "day", "base", "B1.0"]].to_csv(W + "g347_{MODE}_m.csv", index=False); print("identity M3 base (A17) against g280 pc4:", float(np.abs(Dm.set_index("w").base - pd.read_csv(W + "g280_m3.csv").set_index("w").pc4.reindex(Dm.w).to_numpy()).max()), flush=True)
    if {CHK is not None}:
        print("identity June onset against g342:", float(np.abs(Dt.set_index("w").L_base - pd.read_csv(W + "g342_jun80_onset.csv").set_index("w").L_base.reindex(Dt.w).to_numpy()).max()), flush=True)
    print("dumped", len(DON_), "onset windows; train-month windows", len(Dt), flush=True); sys.exit()
    Dt = None''')
    exec(compile(s[0], "g170_g347", "exec"))
else:
    s = [open(R0 + "review/g328_oracle_map.py", encoding="utf-8").read()]; sub = subber(s)
    sub('    for pr, p, ex, nr, _ in v["k"].values():\n        r["base"] += pr * G.iou(g132.gate(bs(tmp(p, 0.8)), ex, nr), m)\n',
        '    for kk_, (pr, p, ex, nr, _) in v["k"].items():\n        q_ = tmp(p, 0.8); s_ = g132.gate(bs(q_), ex, nr); r["base"] += pr * G.iou(s_, m)\n'
        '        if pr > DOG_.get(w, (-1,))[0]:\n            DOG_[w] = (pr, kk_, q_.astype(np.float32), np.asarray(s_).copy(), lq.copy(), m["y"].T.astype(bool), m["m"].T.astype(bool), m["panel"], int(m["t"]), int(m["day"]), "validation")\n')
    sub('                    r["base"] += pr * h.iou(h.far_gate(c, h.decode(tmp(pb, 0.8)), t, kk, lq), y, e, "queue_ongoing")\n',
        '                    q_ = tmp(pb, 0.8); s_ = h.far_gate(c, h.decode(q_), t, kk, lq); r["base"] += pr * h.iou(s_, y, e, "queue_ongoing")\n'
        '                    if pr > DOG_.get(int(k), (-1,))[0]:\n                        DOG_[int(k)] = (pr, kk, q_.astype(np.float32), np.asarray(s_).copy(), lq.copy(), y.copy(), e.copy(), p, int(t), int(dom), "train")\n')
    sub("    D = pd.DataFrame(rows); b = D.base;", f"    D = pd.DataFrame(rows); D.to_csv(W + 'g347_{MODE}_' + lab.split(' (')[0].replace(' / ', '_').replace(' ', '_') + '.csv', index=False); b = D.base;")
    sub("D4 = pickle.load(open(W + \"g284_og_pretemp_dev.pkl\", \"rb\")); rows = []", "DOG_ = {}; D4 = pickle.load(open(W + \"g284_og_pretemp_dev.pkl\", \"rb\")); rows = []")
    if MODE == "og_jun80":
        s[0] = s[0].replace("junmh0", "jun80").replace("g321_jun_", "g342_jun80_").replace("g329_jun_onset_rows", "g342_jun80_onset_rows")
        sub('(("January / February", ("2031-01", "2031-02")), ("June 2030", ("2030-06",)))', '(("June 2030", ("2030-06",)),)')
    else:
        sub('(("January / February", ("2031-01", "2031-02")), ("June 2030", ("2030-06",)))', '(("January / February", ("2031-01", "2031-02")),)')
    s[0] += f'\npickle.dump(DOG_, open(W + "g347_{MODE}.pkl", "wb")); print("dumped", len(DOG_), "ongoing windows", flush=True)\n'
    exec(compile(s[0], "g328_g347", "exec"))
