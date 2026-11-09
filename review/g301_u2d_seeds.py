"""G301 (11/05, ledger #240b, measured before any GPU training): the 2D U-Net's seed count inside A17's March ongoing mix on M3 v3 dev
(G212's frame as G276 / G284, 356 windows, hold-out days not read): the mix with pu = u2d s0 alone, s1 alone and both, at A17's temperature
0.8 -> best_set -> far gate; d = IoU(2) - mean IoU(1), the c / n law predicts 2 -> 4 as d / 2 (line +0.0005 IoU, #233 / #232).
Window bootstrap SE (2000). Identity: both seeds at T 1.2 = G276's a12. usage: TFB_X3=x3af TFB_M3VER=v3 py -B -X utf8 g301_u2d_seeds.py dev"""
import sys
assert sys.argv[1:] == ["dev"], "dev only"
R_ = r"C:/Users/user/tfb/review/"; END = 'exec(compile(s, "g175a", "exec"))'
src = open(R_ + "g212_a10_read.py", encoding="utf-8").read(); assert src.count(END) == 1
exec(compile(src.replace(END, ""), "g212", "exec"))
F_ = lambda u: '(0.8 * (0.8 * ((1 - X3) * (0.8 * xs + 0.2 * pt) + X3 * px) + 0.2 * ' + u + ') + 0.2 * pl) * cf * sw'
t8 = lambda e: 'sg(lg(' + e + ') / 0.8)'
sub('sc["a12"] += pr * G.iou(dec(tmp(' + F_("pu") + ')), m)',
    'sc["a12"] += pr * G.iou(dec(tmp(' + F_("pu") + ')), m); sc["u2"] += pr * G.iou(dec(' + t8(F_("pu")) + '), m); '
    'sc["u0"] += pr * G.iou(dec(' + t8(F_("PU[0][w]")) + '), m); sc["u1"] += pr * G.iou(dec(' + t8(F_("PU[1][w]")) + '), m)')
sub('"a12", "b10"); cols', '"a12", "b10", "u2", "u0", "u1"); cols')
sub('D.to_csv(W + "g212_dev.csv"', 'D.to_csv(W + "g301_dev.csv"')
sub('    Df = train_B(); Df.to_csv(W + "g212_train.csv", index=False)\n', '''    R6 = pd.read_csv(W + "g276_dev.csv").set_index("w").a12; print("identity a12 against g276:", float(np.abs(D.set_index("w").a12 - R6.reindex(D.w).to_numpy()).max()), flush=True)
    R7 = pd.read_csv(W + "g287_m3.csv").set_index("w")["t0.8"]; print("identity u2 (A17) against g287 t0.8:", float(np.abs(D.set_index("w").u2 - R7.reindex(D.w).to_numpy()).max()), flush=True)
    d_ = (D.u2 - (D.u0 + D.u1) / 2).to_numpy(); rg_ = np.random.default_rng(0); se_ = float(np.std([d_[rg_.integers(0, len(d_), len(d_))].mean() for _ in range(2000)]))
    print(f"u2d 1 -> 2 seeds inside A17 (M3 dev ongoing {len(D)}): d {d_.mean():+.5f} (SE {se_:.5f}) -> predicted 2 -> 4 {d_.mean() / 2:+.5f} (SE {se_ / 2:.5f}) | line +0.0005: {d_.mean() / 2 >= 0.0005} | windows that differ {int((np.abs(d_) > 1e-12).sum())}", flush=True)
    sys.exit()
''')
exec(compile(s, "g175s", "exec"))
