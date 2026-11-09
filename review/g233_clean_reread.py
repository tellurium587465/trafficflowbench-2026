"""G233 (11/03, ledger #201): components closed on the masked-history instrument (M3 dev as a whole), read again on the clean ones
(line revision 2): M3 dev's top third of history coverage (nn_test_m3v3/R.npy[:, :12, 2] > the dev 2/3 quantile, as G212; April
composition, m3v2.summary) and the 316 January / February selector windows (unmasked history; paired mean, cluster month x panel x
day). Merged = inverse-variance mean of the two. Saved per-window results only (G181, G197); nothing is recomputed.
  plt  : #134 ongoing Platt per horizon row          (G181: A -> plt; train SCG -> plt)
  s50  : #135 far gate x0.5 (soft) in place of hard  (G181: A -> s50; the January / February plus equals "no gate", so the top third only)
  u2dA : #133 2D U-Net 0.2 on A8's flow              (G197: bundle -> bu; train on B9s' flow Bc -> Bu, the train-month analogue)
  u2dB : #133 on B9s' flow                           (G197: Bc -> Bu)
usage: TFB_M3VER=v3 py -B -X utf8 g233_clean_reread.py"""
import os, sys
os.environ.setdefault("TFB_M3VER", "v3")
import numpy as np, pandas as pd
sys.path.insert(0, r"C:/Users/user/tfb/src")
import m3v2
W = "C:/Users/user/tfb/work/t2/"
d_ = "C:/tfb_data/tfb_nn/nn_test_m3v3/"; R_ = np.load(d_ + "R.npy", mmap_mode="r"); o_ = np.load(d_ + "off.npy"); w_ = [str(x) for x in np.load(d_ + "window_id.npy", allow_pickle=True)]
COV = {x: float(np.asarray(R_[o_[i]:o_[i + 1], :12, 2], float).mean()) for i, x in enumerate(w_)}


def boot(d, cl, n_boot=2000, seed=0):
    rng = np.random.default_rng(seed); ids = pd.factorize(cl)[0]; M = np.bincount(ids, d); N = np.bincount(ids)
    Rm = rng.multinomial(len(M), np.full(len(M), 1.0 / len(M)), size=n_boot); return float(((Rm @ M) / (Rm @ N)).std())


ITEMS = [("plt", "g181_m3_og.csv", "A", "plt", "g181_train_og.csv", "SCG", "plt"),
         ("s50", "g181_m3_og.csv", "A", "s50", None, None, None),
         ("u2dA", "g197_dev.csv", "bundle", "bu", "g197_train.csv", "Bc", "Bu"),
         ("u2dB", "g197_dev.csv", "Bc", "Bu", "g197_train.csv", "Bc", "Bu")]
for nm, fd, a, b, ft, ta, tb in ITEMS:
    D = pd.read_csv(W + fd); D["cov"] = D.w.astype(str).map(COV); qe = D["cov"].quantile(2 / 3); top = D[D["cov"] > qe].reset_index(drop=True)
    full = m3v2.summary(D, a, b, "private"); s3 = m3v2.summary(top, a, b, "private", n_boot=2000, n_draw=500)
    parts = [(s3["t2_weighted"], s3["se"])]; out = [f"M3 all {full['t2_weighted']:+.4f} (SE {full['se']:.4f}) | top third ({len(top)}) {s3['t2_weighted']:+.4f} (SE {s3['se']:.4f})"]
    if ft:
        T = pd.read_csv(W + ft); d = (T[tb] - T[ta]).to_numpy(); cl = (T.month.astype(str) + "_" + T.panel.astype(str) + "_" + T.day.astype(str)).to_numpy(); se = boot(d, cl)
        parts.append((float(d.mean()), se)); out.append(f"Jan/Feb ({len(T)}) {d.mean():+.4f} (SE {se:.4f}, z {d.mean() / max(se, 1e-12):+.2f})")
    wts = [1 / s ** 2 for _, s in parts]; m = sum(w * x for w, (x, _) in zip(wts, parts)) / sum(wts); sem = sum(wts) ** -0.5
    jf_ok = (parts[1][0] / max(parts[1][1], 1e-12) >= -1.64) if ft else True
    print(f"{nm}: " + " | ".join(out) + f" | clean merged {m:+.4f} (SE {sem:.4f}, z {m / sem:+.2f}) | revision-2 line (merged >= 0 and Jan/Feb z >= -1.64): {'met' if m >= 0 and jf_ok else 'not met'}", flush=True)
