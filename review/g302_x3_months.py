"""G302 (11/05, ledger #241 stage 0 (2)): the effect of the newest training months on a component with the same recipe, saved probabilities
only. Inside A17's March ongoing mix on M3 v3 dev (G212's frame as G301, 356 windows, temperature 0.8 -> best_set -> far gate), the raw-history
network x3af (FINAL = nine months 2030-06 .. 2031-02) against x3a (the same configuration with FINAL=0, fold 2030-12 = six months without
2030-12 .. 2031-02; probs_m3v3cap__x3a_s*_fold2030-12). Paired difference, window bootstrap SE (2000) and m3v2.summary (private weights).
Identity: x3af at T 0.8 = G287's t0.8. usage: TFB_X3=x3af TFB_M3VER=v3 py -B -X utf8 g302_x3_months.py dev"""
import sys
assert sys.argv[1:] == ["dev"], "dev only"
R_ = r"C:/Users/user/tfb/review/"; END = 'exec(compile(s, "g175a", "exec"))'
src = open(R_ + "g212_a10_read.py", encoding="utf-8").read(); assert src.count(END) == 1
exec(compile(src.replace(END, ""), "g212", "exec"))
F_ = lambda x: '(0.8 * (0.8 * ((1 - X3) * (0.8 * xs + 0.2 * pt) + X3 * ' + x + ') + 0.2 * pu) + 0.2 * pl) * cf * sw'
t8 = lambda e: 'sg(lg(' + e + ') / 0.8)'
sub('sc["a12"] += pr * G.iou(dec(tmp(' + F_("px") + ')), m)',
    'sc["a12"] += pr * G.iou(dec(tmp(' + F_("px") + ')), m); sc["x9"] += pr * G.iou(dec(' + t8(F_("px")) + '), m); sc["x6"] += pr * G.iou(dec(' + t8(F_("(PX6[0][w] + PX6[1][w]) / 2")) + '), m)')
sub('"a12", "b10"); cols', '"a12", "b10", "x9", "x6"); cols')
sub('def frame(wins, parts):', 'PX6 = [pickle.load(open(W + f"probs_m3v3cap__x3a_s{s_}_fold2030-12.pkl", "rb")) for s_ in (0, 1)]\n\n\ndef frame(wins, parts):')
sub('D.to_csv(W + "g212_dev.csv"', 'D.to_csv(W + "g302_dev.csv"')
sub('    Df = train_B(); Df.to_csv(W + "g212_train.csv", index=False)\n', '''    R7 = pd.read_csv(W + "g287_m3.csv").set_index("w")["t0.8"]; print("identity x9 against g287 t0.8:", float(np.abs(D.set_index("w").x9 - R7.reindex(D.w).to_numpy()).max()), flush=True)
    d_ = (D.x9 - D.x6).to_numpy(); rg_ = np.random.default_rng(0); se_ = float(np.std([d_[rg_.integers(0, len(d_), len(d_))].mean() for _ in range(2000)])); sm = m3v2.summary(D, "x6", "x9", "private")
    print(f"x3af (9 months) - x3a (6 months) inside A17, M3 dev ongoing {len(D)}: {d_.mean():+.5f} (SE {se_:.5f}, z {d_.mean() / max(se_, 1e-12):+.2f}) | private-weighted {sm['t2_weighted']:+.5f} (SE {sm['se']:.5f}, z {sm['t2_weighted'] / max(sm['se'], 1e-12):+.2f}) | windows that differ {int((np.abs(d_) > 1e-12).sum())}", flush=True)
    sys.exit()
''')
exec(compile(s, "g175s", "exec"))
