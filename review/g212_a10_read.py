"""G212 (11/02, ledger #179): the read of A10 against A7 and of B9s against B7 on the ongoing windows. G175's source with one column more: a10 = A8's flow with the same-day X1 of the row-T networks q24t_attnglob s0, s1 in place of c24f's; a10nt = the same without tfm4. dev : M3 dev, writes g212_dev.csv. read : hold-out ongoing only, one read on 11/4 after #63, needs TFB_READ_1104=1, merges each side with its dev value by inverse variance and prints the line of #179 (merged point >= 0; January, February and the LB direction from the ledger). G175 and its csv are not changed.
usage: TFB_X3=x3af TFB_M3VER=v3 py -B -X utf8 g212_a10_read.py dev | TFB_READ_1104=1 ... read"""
import sys
s = open(r"C:/Users/user/tfb/review/g175_bundle_read.py", encoding="utf-8").read()


def sub(a, b, n=1):
    global s
    assert s.count(a) == n, (a[:60], s.count(a))
    s = s.replace(a, b)


sub('SD = {}\n', 'SD = {}; SDT = {}\n')
sub('    CC = ("A", "bundle", "B", "Bc", "B2", "Bc2"); cols', '    for s in (0, 1):\n        for f in glob.glob(W + f"x1_arms/{parts}/sd__q24t_attnglob_s{s}__*.pkl"):\n            for key, v in pickle.load(open(f, "rb")).items():\n                SDT.setdefault(key, []).append(v)\n    CC = ("A", "bundle", "B", "Bc", "B2", "Bc2", "a10", "a10nt"); cols')
sub('sc["bundle"] += pr * G.iou(dec(tmp((0.8 * ((1 - X3) * (0.8 * xs + 0.2 * pt) + X3 * px) + 0.2 * pl) * cf * sw)), m)', 'sc["bundle"] += pr * G.iou(dec(tmp((0.8 * ((1 - X3) * (0.8 * xs + 0.2 * pt) + X3 * px) + 0.2 * pl) * cf * sw)), m); xt = sum(SDT[(w, k)]) / 2; sc["a10"] += pr * G.iou(dec(tmp((0.8 * ((1 - X3) * (0.8 * xt + 0.2 * pt) + X3 * px) + 0.2 * pl) * cf * sw)), m); sc["a10nt"] += pr * G.iou(dec(tmp((0.8 * ((1 - X3) * xt + X3 * px) + 0.2 * pl) * cf * sw)), m)')
sub('"g175_dev.csv"', '"g212_dev.csv"', 2); sub('"g175_hold.csv"', '"g212_hold.csv"'); sub('"g175_train_B.csv"', '"g212_train.csv"')
sub('for a, b, nm in (("A", "bundle", "A side"), ("B", "Bc", "B side"), ("B2", "Bc2", "B side with X1 beta 0.25")):\n        s = m3v2.summary(D, a, b, "private"); sm =', 'for a, b, nm in (("A", "bundle", "A8 against A7"), ("A", "a10", "A10 against A7"), ("bundle", "a10", "A10 against A8"), ("a10nt", "a10", "A10 against A10 without tfm4"), ("B", "Bc", "B9s against B7")):\n        s = m3v2.summary(D, a, b, "private"); sm =')
sub('for a, b, nm in (("A", "bundle", "A side"), ("B", "Bc", "B side"), ("B2", "Bc2", "B side with X1 beta 0.25")):\n    s = m3v2.summary(D, a, b, "private"); d =', 'for a, b, nm in (("A", "a10", "A10 against A7"), ("A", "bundle", "A8 against A7"), ("a10nt", "a10", "A10 against A10 without tfm4 (#63 on A10: z <= -2 drops tfm4)"), ("B", "Bc", "B9s against B7")):\n    s = m3v2.summary(D, a, b, "private"); d =')
sub('/1.25 {mg / se / 1.25:+.2f}) | line /1.25 >= +2: {mg / se / 1.25 >= 2}', '/1.25 {mg / se / 1.25:+.2f}) | line #179, merged point >= 0: {mg >= 0}')
sub('    s = m3v2.summary(D, a, b, "private"); d = m3v2.summary(Dv, a, b, "private");', '    cov_ = COV(); D["cov"] = D.w.map(cov_); Dv["cov"] = Dv.w.map(cov_); qe = Dv["cov"].quantile(2 / 3); s3 = m3v2.summary(D[D["cov"] > qe].reset_index(drop=True), a, b, "private", n_boot=1000, n_draw=500); d3 = m3v2.summary(Dv[Dv["cov"] > qe].reset_index(drop=True), a, b, "private", n_boot=1000, n_draw=500); print(f"{nm}: top coverage third (history visible > {qe:.3f}): hold-out {s3[\'t2_weighted\']:+.4f} (SE {s3[\'se\']:.4f}) | dev {d3[\'t2_weighted\']:+.4f} (SE {d3[\'se\']:.4f})", flush=True)\n    s = m3v2.summary(D, a, b, "private"); d = m3v2.summary(Dv, a, b, "private");')
sub('def frame(wins, parts):', 'def COV():\n    d_ ="C:/tfb_data/tfb_nn/nn_test_m3v3/"; R_ = np.load(d_ + "R.npy", mmap_mode="r"); o_ = np.load(d_ + "off.npy"); w_ = [str(x) for x in np.load(d_ + "window_id.npy", allow_pickle=True)]; return {x: float(np.asarray(R_[o_[i]:o_[i + 1], :12, 2], float).mean()) for i, x in enumerate(w_)}\n\n\ndef frame(wins, parts):')
sub('("A", "a10", "A10 against A7"), ("A", "bundle", "A8 against A7"), ("a10nt", "a10", "A10 against A10 without tfm4 (#63 on A10: z <= -2 drops tfm4)"), ("B", "Bc", "B9s against B7"))', '("A", "bundle", "A8 against A7 (main)"), ("B", "Bc", "B9s against B7 (main)"), ("A", "a10", "A10 against A7 (reference)"), ("bundle", "a10", "A10 against A8 (reference)"))')
# 11/03 ledgers #179, #201: the defaults A12 / B10. a12 = A8's flow + the 2D U-Net at 0.2 right after the X3a mix (G197's "bu"), b10 = B9s' flow + the
# same (G197's "Bu"); #197 (row T as the last queue) changes 1 official window and is left out of this ongoing read, #17 is onset (read elsewhere)
sub('CC = ("A", "bundle", "B", "Bc", "B2", "Bc2", "a10", "a10nt"); cols', 'PU = [pickle.load(open(W + f"probs_m3v3__u2d_s{s_}.pkl", "rb")) for s_ in (0, 1)]; CC = ("A", "bundle", "B", "Bc", "B2", "Bc2", "a10", "a10nt", "a12", "b10"); cols')
sub('sc["a10nt"] += pr * G.iou(dec(tmp((0.8 * ((1 - X3) * xt + X3 * px) + 0.2 * pl) * cf * sw)), m)', 'sc["a10nt"] += pr * G.iou(dec(tmp((0.8 * ((1 - X3) * xt + X3 * px) + 0.2 * pl) * cf * sw)), m); pu = (PU[0][w] + PU[1][w]) / 2; sc["a12"] += pr * G.iou(dec(tmp((0.8 * (0.8 * ((1 - X3) * (0.8 * xs + 0.2 * pt) + X3 * px) + 0.2 * pu) + 0.2 * pl) * cf * sw)), m)')
sub('sc["Bc"] += pr * G.iou(raw(tmp((1 - X3) * pq + X3 * px)), m);', 'sc["Bc"] += pr * G.iou(raw(tmp((1 - X3) * pq + X3 * px)), m); sc["b10"] += pr * G.iou(raw(tmp(0.8 * ((1 - X3) * pq + X3 * px) + 0.2 * (PU[0][w] + PU[1][w]) / 2)), m);')
sub('for a, b, nm in (("A", "bundle", "A8 against A7"), ("A", "a10", "A10 against A7"), ("bundle", "a10", "A10 against A8"), ("a10nt", "a10", "A10 against A10 without tfm4"), ("B", "Bc", "B9s against B7")):',
    'for a, b, nm in (("A", "a12", "A12 against A7"), ("B", "b10", "B10 against B7"), ("bundle", "a12", "A12 against A8"), ("Bc", "b10", "B10 against B9s"), ("A", "bundle", "A8 against A7"), ("B", "Bc", "B9s against B7"), ("A", "a10", "A10 against A7")):')
sub('("A", "bundle", "A8 against A7 (main)"), ("B", "Bc", "B9s against B7 (main)"), ("A", "a10", "A10 against A7 (reference)"), ("bundle", "a10", "A10 against A8 (reference)"))',
    '("A", "a12", "A12 against A7 (main)"), ("B", "b10", "B10 against B7 (main)"), ("A", "bundle", "A8 against A7"), ("B", "Bc", "B9s against B7"), ("A", "a10", "A10 against A7 (reference)"))')
exec(compile(s, "g175a", "exec"))
