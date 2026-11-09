"""G249 (11/04, ledger #212): the read of G248. G212's dev frame (M3 v3 dev ongoing 356 windows, A12's flow "a12") with one column
more: a12s = the same flow with the same-day X1 of c24f_attnglob s0, s1 averaged over the fine-tune's seeds 0 and 1 (G141's sd parts +
G248's sd1 parts) in place of seed 0 alone. Dev only (no read mode here); G212, its csv and their sha lines are not changed (writes
g249_dev.csv). Prints seeds 0+1 minus seed 0 on all dev windows and on the top third of history coverage (G212's definition, April
composition), and |dp| between the seeds per (window, k).
usage: TFB_X3=x3af TFB_M3VER=v3 py -B -X utf8 g249_x1_seed_read.py dev"""
import sys
assert sys.argv[1:] == ["dev"], "dev only"
R_ = r"C:/Users/user/tfb/review/"; END = 'exec(compile(s, "g175a", "exec"))'
src = open(R_ + "g212_a10_read.py", encoding="utf-8").read(); assert src.count(END) == 1
exec(compile(src.replace(END, ""), "g212", "exec"))   # G175's source with G212's columns in s, nothing run
sub('SD = {}; SDT = {}\n', 'SD = {}; SDT = {}; SD1 = {}\n')
sub('    PU = [pickle.load(', '    for s in (0, 1):\n        for f in glob.glob(W + f"x1_arms/{parts}/sd1__c24f_attnglob_s{s}__*.pkl"):\n            for key, v in pickle.load(open(f, "rb")).items():\n                SD1.setdefault(key, []).append(v)\n'
    '    assert set(SD1) == set(SD) and all(len(v) == 2 for v in SD1.values()), "G248 parts incomplete"\n    PU = [pickle.load(')
sub('"a12", "b10"); cols', '"a12", "b10", "a12s"); cols')
sub('sc["a12"] += pr * G.iou(dec(tmp((0.8 * (0.8 * ((1 - X3) * (0.8 * xs + 0.2 * pt) + X3 * px) + 0.2 * pu) + 0.2 * pl) * cf * sw)), m)',
    'sc["a12"] += pr * G.iou(dec(tmp((0.8 * (0.8 * ((1 - X3) * (0.8 * xs + 0.2 * pt) + X3 * px) + 0.2 * pu) + 0.2 * pl) * cf * sw)), m); '
    'xs1 = sum(SD1[(w, k)]) / 2; DP.append(float(np.abs(xs1 - xs).mean())); xq = (xs + xs1) / 2; sc["a12s"] += pr * G.iou(dec(tmp((0.8 * (0.8 * ((1 - X3) * (0.8 * xq + 0.2 * pt) + X3 * px) + 0.2 * pu) + 0.2 * pl) * cf * sw)), m)')
sub('D.to_csv(W + "g212_dev.csv"', 'D.to_csv(W + "g249_dev.csv"')
sub('    Df = train_B(); Df.to_csv(W + "g212_train.csv", index=False)\n',
    '    cov_ = COV(); D["cov"] = D.w.map(cov_); top = D[D["cov"] > D["cov"].quantile(2 / 3)].reset_index(drop=True)\n'
    '    for nm_, d_ in (("all", D), ("top coverage third", top)):\n'
    '        s_ = m3v2.summary(d_, "a12", "a12s", "private", n_boot=2000, n_draw=500); print(f"seeds 0+1 minus seed 0 ({nm_}, {len(d_)} windows): {s_[\'t2_weighted\']:+.4f} (SE {s_[\'se\']:.4f}, z {s_[\'t2_weighted\'] / max(s_[\'se\'], 1e-12):+.2f}) | windows changed {int((~np.isclose(d_.a12s, d_.a12)).sum())} | raw {(d_.a12s - d_.a12).mean():+.4f}", flush=True)\n'
    '    print(f"|dp| between the seeds per (window, k): mean {np.mean(DP):.4f}, median {np.median(DP):.4f}, p90 {np.quantile(DP, 0.9):.4f}, max {np.max(DP):.4f}", flush=True)\n'
    '    sys.exit()\n')
s = "DP = []\n" + s
exec(compile(s, "g175c", "exec"))
