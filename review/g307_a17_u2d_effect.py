"""G307 (11/05, ledger #245 (1)): u2d's effect on A17 on M3 v3 dev ongoing (G212's frame as G301, 356 windows, hold-out days not read):
A17's March ongoing mix at temperature 0.8 with u2d (0.8 (...) + 0.2 u2d, as built) against A21's (the u2d step left out:
0.8 ((1 - X3) (0.8 X1 + 0.2 tfm4) + X3 x3af) + 0.2 LightGBM), best_set -> far gate, k expectation. m3v2.summary (private weights, cluster
bootstrap) on all windows and on the top third of history coverage (G233's instrument). Identity: A17 = G287's t0.8.
usage: TFB_X3=x3af TFB_M3VER=v3 py -B -X utf8 g307_a17_u2d_effect.py dev"""
import sys
assert sys.argv[1:] == ["dev"], "dev only"
R_ = r"C:/Users/user/tfb/review/"; END = 'exec(compile(s, "g175a", "exec"))'
src = open(R_ + "g212_a10_read.py", encoding="utf-8").read(); assert src.count(END) == 1
exec(compile(src.replace(END, ""), "g212", "exec"))
F_ = '(0.8 * (0.8 * ((1 - X3) * (0.8 * xs + 0.2 * pt) + X3 * px) + 0.2 * pu) + 0.2 * pl) * cf * sw'
N_ = '(0.8 * ((1 - X3) * (0.8 * xs + 0.2 * pt) + X3 * px) + 0.2 * pl) * cf * sw'
t8 = lambda e: 'sg(lg(' + e + ') / 0.8)'
sub('sc["a12"] += pr * G.iou(dec(tmp(' + F_ + ')), m)', 'sc["a12"] += pr * G.iou(dec(tmp(' + F_ + ')), m); sc["a17"] += pr * G.iou(dec(' + t8(F_) + '), m); sc["a21"] += pr * G.iou(dec(' + t8(N_) + '), m)')
sub('"a12", "b10"); cols', '"a12", "b10", "a17", "a21"); cols')
sub('D.to_csv(W + "g212_dev.csv"', 'D.to_csv(W + "g307_dev.csv"')
sub('    Df = train_B(); Df.to_csv(W + "g212_train.csv", index=False)\n', '''    R7 = pd.read_csv(W + "g287_m3.csv").set_index("w")["t0.8"]; print("identity a17 against g287 t0.8:", float(np.abs(D.set_index("w").a17 - R7.reindex(D.w).to_numpy()).max()), flush=True)
    cv_ = COV(); D["cov"] = D.w.map(cv_); top = D[D["cov"] > D["cov"].quantile(2 / 3)].reset_index(drop=True)
    for lab, d_ in (("all", D), ("top third of coverage", top)):
        sm = m3v2.summary(d_, "a21", "a17", "private"); print(f"u2d effect on A17 (A17 - A21), M3 dev ongoing {lab} {len(d_)}: {sm['t2_weighted']:+.5f} (SE {sm['se']:.5f}, z {sm['t2_weighted'] / max(sm['se'], 1e-12):+.2f}) | raw mean {(d_.a17 - d_.a21).mean():+.5f} | windows that differ {int((~np.isclose(d_.a17, d_.a21)).sum())}", flush=True)
    sys.exit()
''')
exec(compile(s, "g175s", "exec"))
