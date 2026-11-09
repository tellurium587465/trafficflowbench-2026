"""G260 (11/04, ledger #222, the strategy's 19:45 request): A13 without the no-memory LightGBM at 0.2 on the ongoing windows (A's April
part since 10/23, #30 / #32). G212's dev frame (M3 v3 dev 356 windows; on dev a13 = a12, no dense X1): a12nl = A12's flow with the
last mix 0.8 (...) + 0.2 pl replaced by (...) alone. Dev only, writes g260_dev.csv. All and the top coverage third (April composition).
usage: TFB_X3=x3af TFB_M3VER=v3 py -B -X utf8 g260_a13_nolgb.py dev"""
import sys
assert sys.argv[1:] == ["dev"], "dev only"
R_ = r"C:/Users/user/tfb/review/"; END = 'exec(compile(s, "g175a", "exec"))'
src = open(R_ + "g212_a10_read.py", encoding="utf-8").read(); assert src.count(END) == 1
exec(compile(src.replace(END, ""), "g212", "exec"))
IN = '(0.8 * ((1 - X3) * (0.8 * xs + 0.2 * pt) + X3 * px) + 0.2 * pu)'
sub('sc["a12"] += pr * G.iou(dec(tmp((0.8 * ' + IN + ' + 0.2 * pl) * cf * sw)), m)',
    'sc["a12"] += pr * G.iou(dec(tmp((0.8 * ' + IN + ' + 0.2 * pl) * cf * sw)), m); sc["a12nl"] += pr * G.iou(dec(tmp(' + IN + ' * cf * sw)), m)')
sub('"a12", "b10"); cols', '"a12", "b10", "a12nl"); cols')
sub('D.to_csv(W + "g212_dev.csv"', 'D.to_csv(W + "g260_dev.csv"')
sub('    Df = train_B(); Df.to_csv(W + "g212_train.csv", index=False)\n',
    '    cov_ = COV(); D["cov"] = D.w.map(cov_); top = D[D["cov"] > D["cov"].quantile(2 / 3)].reset_index(drop=True)\n'
    '    for nm_, d_ in (("all", D), ("top coverage third", top)):\n'
    '        s_ = m3v2.summary(d_, "a12", "a12nl", "private", n_boot=2000, n_draw=500); print(f"A13 without lgb minus A13 ({nm_}, {len(d_)}): {s_[\'t2_weighted\']:+.4f} (SE {s_[\'se\']:.4f}, z {s_[\'t2_weighted\'] / max(s_[\'se\'], 1e-12):+.2f}) | windows changed {int((~np.isclose(d_.a12nl, d_.a12)).sum())}", flush=True)\n'
    '    sys.exit()\n')
exec(compile(s, "g175n", "exec"))
