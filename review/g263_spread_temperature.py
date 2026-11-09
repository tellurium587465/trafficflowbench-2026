"""G263 (11/04, ledger #224, report material, no decision): M3 v3 dev ongoing 356 windows. Spread of the 40 ongoing networks (q24f
x 5 configurations x 8 seeds, saved M3 probabilities): per window the mean over cells of the variance across the 40 nets. By thirds of
the spread: IoU of A13's flow (G212's a12; on dev a13 = a12) at the temperature 1.0 / 1.2 (production) / 1.4 before best_set, and the
best temperature per third. Dev only (no hold-out); G212's frame with two columns more, writes g263_dev.csv.
usage: TFB_X3=x3af TFB_M3VER=v3 py -B -X utf8 g263_spread_temperature.py dev"""
import sys
assert sys.argv[1:] == ["dev"], "dev only"
R_ = r"C:/Users/user/tfb/review/"; END = 'exec(compile(s, "g175a", "exec"))'
src = open(R_ + "g212_a10_read.py", encoding="utf-8").read(); assert src.count(END) == 1
exec(compile(src.replace(END, ""), "g212", "exec"))
FA = '(0.8 * (0.8 * ((1 - X3) * (0.8 * xs + 0.2 * pt) + X3 * px) + 0.2 * pu) + 0.2 * pl) * cf * sw'
sub('sc["a12"] += pr * G.iou(dec(tmp(' + FA + ')), m)',
    'sc["a12"] += pr * G.iou(dec(tmp(' + FA + ')), m); sc["a12t10"] += pr * G.iou(dec(' + FA + '), m); sc["a12t14"] += pr * G.iou(dec(sg(lg(' + FA + ') / 1.4)), m)')
sub('"a12", "b10"); cols', '"a12", "b10", "a12t10", "a12t14"); cols')
sub('D.to_csv(W + "g212_dev.csv"', 'D.to_csv(W + "g263_dev.csv"')
sub('    Df = train_B(); Df.to_csv(W + "g212_train.csv", index=False)\n',
    '    NETS = [f"q24f_{c}_s{s_}" for s_ in range(8) for c in ("attnglob", "attnwp05", "attn1", "drop1", "r13")]; PP = [pickle.load(open(W + f"probs_m3v3__{t_}.pkl", "rb")) for t_ in NETS]\n'
    '    D["spread"] = [float(np.var(np.stack([d_[w_] for d_ in PP]), axis=0).mean()) for w_ in D.w]; D.to_csv(W + "g263_dev.csv", index=False)\n'
    '    D["q"] = pd.qcut(D.spread.rank(method="first"), 3, labels=["low", "mid", "high"])\n'
    '    for q_, d_ in D.groupby("q", observed=True):\n'
    '        lv = {te: d_[c_].mean() for te, c_ in ((1.0, "a12t10"), (1.2, "a12"), (1.4, "a12t14"))}; bte = max(lv, key=lv.get)\n'
    '        s0 = m3v2.summary(d_.reset_index(drop=True), "a12", "a12t10", "private", n_boot=1000, n_draw=300); s4 = m3v2.summary(d_.reset_index(drop=True), "a12", "a12t14", "private", n_boot=1000, n_draw=300)\n'
    '        print(f"spread {q_} ({len(d_)} windows, {d_.spread.min():.4f}-{d_.spread.max():.4f}): IoU T1.0 {lv[1.0]:.4f} T1.2 {lv[1.2]:.4f} T1.4 {lv[1.4]:.4f} | best T {bte} | judge T1.0 - T1.2 {s0[\'t2_weighted\']:+.4f} (SE {s0[\'se\']:.4f}), T1.4 - T1.2 {s4[\'t2_weighted\']:+.4f} (SE {s4[\'se\']:.4f})", flush=True)\n'
    '    print("corr(spread, a12)", round(D.spread.corr(D.a12), 3), flush=True)\n'
    '    sys.exit()\n')
exec(compile(s, "g175t", "exec"))
