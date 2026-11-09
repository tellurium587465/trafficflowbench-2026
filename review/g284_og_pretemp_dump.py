"""G284 (11/05, ledger #234 data, before any temperature is chosen): A's March ongoing flow on M3 v3 dev (G212's frame, the a12 column =
A16's March ongoing: 0.8 (0.8 ((1 - X3) (0.8 X1 + 0.2 tfm4) + X3 x3af) + 0.2 u2d) + 0.2 LightGBM, times the cone and the seen x0.85)
written out just before the temperature, per (window, k) with the k weight (11/05 #234 revision: also G119's capB0 flow, the mean of
capB0 s0-3 times the same cone and seen x0.85, for the instrument shift delta), plus the window's labels and scored-cell mask.
Identity: the IoU of the dumped probabilities at temperature 1.2 through the same decode must equal G276's a12 column (g276_dev.csv).
Writes work/t2/g284_og_pretemp_dev.pkl. Dev only. usage: TFB_X3=x3af TFB_M3VER=v3 py -B -X utf8 g284_og_pretemp_dump.py dev"""
import sys, pickle
assert sys.argv[1:] == ["dev"], "dev only"
R_ = r"C:/Users/user/tfb/review/"; END = 'exec(compile(s, "g175a", "exec"))'
src = open(R_ + "g212_a10_read.py", encoding="utf-8").read(); assert src.count(END) == 1
exec(compile(src.replace(END, ""), "g212", "exec"))
F_ = '(0.8 * (0.8 * ((1 - X3) * (0.8 * xs + 0.2 * pt) + X3 * px) + 0.2 * pu) + 0.2 * pl) * cf * sw'
sub('sc["a12"] += pr * G.iou(dec(tmp(' + F_ + ')), m)',
    'sc["a12"] += pr * G.iou(dec(tmp(' + F_ + ')), m); DUMP.setdefault(w, dict(y=m["y"], m=m["m"], k={}))["k"][k] = (pr, (' + F_ + '), ex, nr, (PCB[w] * cf * sw))')
sub('def frame(wins, parts):', 'DUMP = {}; _PC = [pickle.load(open(W + f"probs_m3v3__capB0_s{s_}_fold2030-12.pkl", "rb")) for s_ in range(4)]; PCB = {w_: sum(d_[w_] for d_ in _PC) / 4 for w_ in _PC[0]}\n\n\ndef frame(wins, parts):')
sub('D.to_csv(W + "g212_dev.csv"', 'pickle.dump(DUMP, open(W + "g284_og_pretemp_dev.pkl", "wb")); D.to_csv(W + "g284_dev.csv"')
sub('    Df = train_B(); Df.to_csv(W + "g212_train.csv", index=False)\n', '    R6 = pd.read_csv(W + "g276_dev.csv").set_index("w").a12; print("windows dumped", len(DUMP), "| identity a12 against g276_dev.csv: max |d|", float(np.abs(D.set_index("w").a12 - R6.reindex(D.w).to_numpy()).max()), flush=True); sys.exit()\n')
exec(compile(s, "g175s", "exec"))
