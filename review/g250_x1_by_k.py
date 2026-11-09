"""G250 (11/04, ledger #213, the strategy's (D) from panel 2): the same-day X1's gain by k (earlier days available, m3v2.kdist with
cap 3 as the read's flow; k = 0, 1, 2, 3+). G212's dev frame (M3 v3 dev ongoing 356 windows, A12's flow "a12"), per (window, k) the
IoU of the flow with the X1 term xb = beta * xs + (1 - beta) * pb for beta = 1 (production), 0.25 and 0, where xs = the same-day X1 of
c24f_attnglob s0, s1 (G141 sd) and pb = the same two networks without the fine-tune (probs_m3v3__c24f_attnglob_s*). Dev only, writes
g250_k.csv; G212 and its csv are not changed. Per k: mean of (beta - 0) over the windows drawn at that k, cluster bootstrap (panel x
day), all dev windows and the top third of history coverage. January / February are not read here (their same-day X1 needs a recompute).
usage: TFB_X3=x3af TFB_M3VER=v3 py -B -X utf8 g250_x1_by_k.py dev"""
import sys
assert sys.argv[1:] == ["dev"], "dev only"
R_ = r"C:/Users/user/tfb/review/"; END = 'exec(compile(s, "g175a", "exec"))'
src = open(R_ + "g212_a10_read.py", encoding="utf-8").read(); assert src.count(END) == 1
exec(compile(src.replace(END, ""), "g212", "exec"))
F_ = lambda x_: '(0.8 * (0.8 * ((1 - X3) * (0.8 * ' + x_ + ' + 0.2 * pt) + X3 * px) + 0.2 * pu) + 0.2 * pl) * cf * sw'
sub('sc["a12"] += pr * G.iou(dec(tmp(' + F_("xs") + ')), m)',
    'sc["a12"] += pr * G.iou(dec(tmp(' + F_("xs") + ')), m); pbm = (PB[0][w] + PB[1][w]) / 2; '
    'KR.append(dict(w=w, k=k, pr=pr, b1=G.iou(dec(tmp(' + F_("xs") + ')), m), b25=G.iou(dec(tmp(' + F_("(0.25 * xs + 0.75 * pbm)") + ')), m), b0=G.iou(dec(tmp(' + F_("pbm") + ')), m)))')
sub('D.to_csv(W + "g212_dev.csv"', 'D.to_csv(W + "g250_dev.csv"')
sub('    Df = train_B(); Df.to_csv(W + "g212_train.csv", index=False)\n',
    '    K = pd.DataFrame(KR); K.to_csv(W + "g250_k.csv", index=False); cov_ = COV(); K["cov"] = K.w.map(cov_); qe = D.w.map(cov_).quantile(2 / 3)\n'
    '    K["cl"] = K.w.map(lambda x: f"{G.META[x][\'panel\']}_{G.META[x][\'day\']}"); assert np.allclose(K.groupby("w").apply(lambda d: (d.pr * d.b1).sum()).reindex(D.w).to_numpy(), D.a12.to_numpy()), "identity a12"\n'
    '    def bt(d, cl, n=2000):\n'
    '        ids = pd.factorize(cl)[0]; M_ = np.bincount(ids, d); N_ = np.bincount(ids); Rm = np.random.default_rng(0).multinomial(len(M_), np.full(len(M_), 1 / len(M_)), size=n); return float(((Rm @ M_) / (Rm @ N_)).std())\n'
    '    for lab, KK in (("all dev", K), ("top coverage third", K[K["cov"] > qe])):\n'
    '        for kk in (0, 1, 2, 3):\n'
    '            d = KK[KK.k == kk]\n'
    '            if len(d):\n'
    '                print(f"{lab} k={kk}: windows {len(d)} | beta 1 - 0 {(d.b1 - d.b0).mean():+.4f} (SE {bt((d.b1 - d.b0).to_numpy(), d.cl):.4f}) | beta 0.25 - 0 {(d.b25 - d.b0).mean():+.4f} (SE {bt((d.b25 - d.b0).to_numpy(), d.cl):.4f}) | level b0 {d.b0.mean():.4f}", flush=True)\n'
    '    sys.exit()\n')
s = "KR = []\n" + s
exec(compile(s, "g175d", "exec"))
