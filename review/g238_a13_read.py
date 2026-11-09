"""G238 (11/03, ledgers #189 / #179): G212 with one column more for the read, moved to 11/5 morning (user's approval 11/03 16:20).
a13 = A12's flow (G212's a12) with the same-day X1 of c24f_attnglob s0, s1 on the March pool on origins every 3 rows (G237,
x1_arms/parts_holdout_d3) in place of G174's (spacing 6). Dev has no dense X1, so on dev a13 = a12 and the A13 line is the hold-out
top coverage third alone (#179: A13t - A12t >= -1 SE -> A13, else A12). G212, its csv and their sha lines are not changed: dev writes
g238_dev.csv (the B training read is left to G212), read takes dev from g212_dev.csv and writes g238_hold.csv. Run this in place of
G212 at the read (one read: every G212 line plus the A13 line).
11/04 (#179 3, 4, 6, #214, #216): b11 = B10's flow with the ongoing family over q24f seeds 0-15 (B11), b12 = the same decoded by the
exact expected IoU (dex.DEX, B12); per (window, k) IoU of a12 / a13 / b10 / b11 / b12 for the k <= 1 / k >= 2 layers (descriptive);
the E[max] table with B11 / B12. The read's tail (TAIL) also runs on the dev frame with G238_SELFTEST=1 in dev mode (no hold-out).
usage: TFB_X3=x3af TFB_M3VER=v3 py -B -X utf8 g238_a13_read.py dev | TFB_READ_1104=1 ... read"""
import sys
R_ = r"C:/Users/user/tfb/review/"; END = 'exec(compile(s, "g175a", "exec"))'
src = open(R_ + "g212_a10_read.py", encoding="utf-8").read(); assert src.count(END) == 1
exec(compile(src.replace(END, ""), "g212", "exec"))   # G175's source with G212's columns in s, nothing run
sub('SD = {}; SDT = {}\n', 'SD = {}; SDT = {}; SD3 = {}\n')
sub('    PU = [pickle.load(', '    for s in (0, 1):\n        for f in glob.glob(W + f"x1_arms/{parts}_d3/sdd3__c24f_attnglob_s{s}__*.pkl"):\n            for key, v in pickle.load(open(f, "rb")).items():\n                SD3.setdefault(key, []).append(v)\n'
    '    assert parts != "parts_holdout" or (set(SD3) == set(SD) and all(len(v) == 2 for v in SD3.values())), "G237 parts incomplete"\n    PU = [pickle.load(')
sub('"a12", "b10"); cols', '"a12", "b10", "a13", "b11", "b12"); cols')
FA = lambda x_: '(0.8 * (0.8 * ((1 - X3) * (0.8 * ' + x_ + ' + 0.2 * pt) + X3 * px) + 0.2 * pu) + 0.2 * pl) * cf * sw'
sub('sc["a12"] += pr * G.iou(dec(tmp(' + FA("xs") + ')), m)',
    'sc["a12"] += pr * G.iou(dec(tmp(' + FA("xs") + ')), m); KV["a12"] = G.iou(dec(tmp(' + FA("xs") + ')), m); '
    'xs3 = sum(SD3[(w, k)]) / 2 if parts == "parts_holdout" else xs; KV["a13"] = G.iou(dec(tmp(' + FA("xs3") + ')), m); sc["a13"] += pr * KV["a13"]')
# 11/04 #179 6: B11 (seeds 0-15) and B12 (B11 decoded by dex.DEX) on B10's flow; per (window, k) records for the k layers
sub('ON = [(f"q24f_{c}_s{s}", w) for s in range(8) for c, w in CF5]; WS = sum(w for _, w in ON)',
    'ON = [(f"q24f_{c}_s{s}", w) for s in range(8) for c, w in CF5]; WS = sum(w for _, w in ON); ON16 = [(f"q24f_{c}_s{s}", w) for s in range(16) for c, w in CF5]; WS16 = sum(w for _, w in ON16)')
sub('PQ = {t: pickle.load(open(W + f"probs_m3v3__{t}.pkl", "rb")) for t, _ in ON}', 'PQ = {t: pickle.load(open(W + f"probs_m3v3__{t}.pkl", "rb")) for t, _ in ON16}')
FB = lambda q_: 'tmp(0.8 * ((1 - X3) * ' + q_ + ' + X3 * px) + 0.2 * (PU[0][w] + PU[1][w]) / 2)'
sub('sc["b10"] += pr * G.iou(raw(' + FB("pq") + '), m);',
    'sc["b10"] += pr * G.iou(raw(' + FB("pq") + '), m); KV["b10"] = G.iou(raw(' + FB("pq") + '), m); pq16 = sum(wt * PQ[t_][w] for t_, wt in ON16) / WS16; '
    'KV["b11"] = G.iou(raw(' + FB("pq16") + '), m); KV["b12"] = G.iou(_dx.DEX(' + FB("pq16") + '), m); sc["b11"] += pr * KV["b11"]; sc["b12"] += pr * KV["b12"]; KR.append(dict(w=w, k=k, pr=pr, **KV));')
sub('D.to_csv(W + "g212_dev.csv"', 'D.to_csv(W + "g238_dev.csv"'); sub('"g212_hold.csv"', '"g238_hold.csv"')
TAIL = ('cov_ = COV(); D["cov"] = D.w.map(cov_); Dv["cov"] = Dv.w.map(cov_); qe = Dv["cov"].quantile(2 / 3); top_ = D[D["cov"] > qe].reset_index(drop=True)\n'
        'for a_, b_, nm_ in (("a12", "a13", "A13 against A12 (main, #189)"), ("b10", "b11", "B11 against B10 (#179 6)"), ("b10", "b12", "B12 against B10 (#179 6)")):\n'
        '    s3 = m3v2.summary(top_, a_, b_, "private", n_boot=1000, n_draw=500); sa = m3v2.summary(D, a_, b_, "private")\n'
        '    print(f"{nm_}: top coverage third ({len(top_)}) {s3[\'t2_weighted\']:+.4f} (SE {s3[\'se\']:.4f}, z {s3[\'t2_weighted\'] / max(s3[\'se\'], 1e-12):+.2f}) | all {len(D)} windows {sa[\'t2_weighted\']:+.4f} (SE {sa[\'se\']:.4f}) | windows changed {int((~np.isclose(D[b_], D[a_])).sum())}", flush=True)\n'
        'sys.path.insert(0, R0 + "review"); from emax import emax_table\n'
        'CE_ = {"A7": (None, "A"), "A8n": (None, "bundle"), "A12": (None, "a12"), "A13": (None, "a13"), "B7": (None, "B"), "B9s": (None, "Bc"), "B10": (None, "b10"), "B11": (None, "b11"), "B12": (None, "b12")}\n'
        'for lab_, g_ in (("all ongoing", D), ("top coverage third", top_)):\n'
        '    T_ = emax_table(None, g_, CE_, [(a_, b_) for a_ in ("A13", "A12", "A8n", "A7") for b_ in ("B12", "B11", "B10", "B9s", "B7")] + [("A13", "A12")], ("A12", "B10"))\n'
        '    print(f"E[max] of the pairs, {lab_} ({len(g_)}), 40-window draws, gain over (A12, B10):", flush=True)\n'
        '    for p_, (lv_, gn_, se_) in sorted(T_.items(), key=lambda x: -x[1][1]):\n'
        '        print(f"  {p_[0]} + {p_[1]}: E[max] {lv_:.5f} gain {gn_:+.5f} (SE {se_:.5f}, z {gn_ / max(se_, 1e-12):+.2f})", flush=True)\n'
        'K_ = pd.DataFrame(KR)\n'
        'def bt_(d, cl, n=2000):\n'
        '    ids = pd.factorize(cl)[0]; M_ = np.bincount(ids, d); N_ = np.bincount(ids); Rm = np.random.default_rng(0).multinomial(len(M_), np.full(len(M_), 1 / len(M_)), size=n); return float(((Rm @ M_) / (Rm @ N_)).std())\n'
        'for lay_, mk_ in (("k <= 1", K_.k <= 1), ("k >= 2", K_.k >= 2)):\n'
        '    g_ = K_[mk_]; ag_ = g_.groupby("w").apply(lambda d: pd.Series({c: float((d.pr * d[c]).sum() / d.pr.sum()) for c in ("a12", "a13", "b10", "b11", "b12")})); cl_ = np.array([f"{G.META[x][\'panel\']}_{G.META[x][\'day\']}" for x in ag_.index])\n'
        '    for a_, b_ in (("a12", "a13"), ("b10", "b11"), ("b10", "b12")):\n'
        '        dd_ = (ag_[b_] - ag_[a_]).to_numpy(); print(f"layer {lay_} (descriptive, #179 4): {b_} - {a_} {dd_.mean():+.4f} (SE {bt_(dd_, cl_):.4f}) over {len(ag_)} windows", flush=True)\n')
sub('    Df = train_B(); Df.to_csv(W + "g212_train.csv", index=False)\n',
    '    if os.environ.get("G238_SELFTEST") == "1":                # the read tail on the dev frame (no hold-out): Dv = D\n'
    '        Dv = D.copy()\n' + ''.join('        ' + l + '\n' for l in TAIL.splitlines()) + '    sys.exit()\n')
s += "\n" + TAIL
s = "import dex as _dx\nKR = []; KV = {}\n" + s
sys.path.insert(0, R_)
exec(compile(s, "g175b", "exec"))
