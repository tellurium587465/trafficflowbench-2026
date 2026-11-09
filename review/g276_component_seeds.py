"""G276 (11/05, ledger #232 measured before any GPU training, declared by the strategy before the results): the seed count of A's
smaller ongoing components inside A's final mix (G212's dev frame, M3 v3 dev 356 windows, A12's flow = A13 on dev, April composition).
  x3af : px from one seed (s0, s1) against both; d = IoU(2) - mean IoU(1); the c / n law predicts 2 -> 4 as d / 2
  tfm4 : pt from every subset of the five seeds (n = 1 .. 5); fit IoU(n) = a - c / n on the means; 5 -> 8 predicted c (1/5 - 1/8)
Window bootstrap SE (300). Dev only. usage: TFB_X3=x3af TFB_M3VER=v3 py -B -X utf8 g276_component_seeds.py dev"""
import sys, itertools
assert sys.argv[1:] == ["dev"], "dev only"
R_ = r"C:/Users/user/tfb/review/"; END = 'exec(compile(s, "g175a", "exec"))'
src = open(R_ + "g212_a10_read.py", encoding="utf-8").read(); assert src.count(END) == 1
exec(compile(src.replace(END, ""), "g212", "exec"))
F_ = lambda xpt, xpx: '(0.8 * (0.8 * ((1 - X3) * (0.8 * xs + 0.2 * ' + xpt + ') + X3 * ' + xpx + ') + 0.2 * pu) + 0.2 * pl) * cf * sw'
SUBS = [c for n in range(1, 6) for c in itertools.combinations(range(5), n)]
sub('sc["a12"] += pr * G.iou(dec(tmp(' + F_("pt", "px") + ')), m)',
    'sc["a12"] += pr * G.iou(dec(tmp(' + F_("pt", "px") + ')), m); '
    'px0_ = PX[0][w]; px1_ = PX[1][w]; sc["x1s0"] += pr * G.iou(dec(tmp(' + F_("pt", "px0_") + ')), m); sc["x1s1"] += pr * G.iou(dec(tmp(' + F_("pt", "px1_") + ')), m); '
    '[sc.__setitem__(f"t{i_}", sc[f"t{i_}"] + pr * G.iou(dec(tmp(' + F_("(sum(PL[j_][w] for j_ in c_) / len(c_))", "px") + ')), m)) for i_, c_ in enumerate(SUBS)]')
sub('"a12", "b10"); cols', '"a12", "b10", "x1s0", "x1s1", ' + ", ".join(f'"t{i}"' for i in range(len(SUBS))) + '); cols')
sub('D.to_csv(W + "g212_dev.csv"', 'D.to_csv(W + "g276_dev.csv"')
sub('    Df = train_B(); Df.to_csv(W + "g212_train.csv", index=False)\n', '    sys.exit()\n')
exec(compile(s, "g175s", "exec"))
