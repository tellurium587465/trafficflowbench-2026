"""G268 (11/04, ledger #226): the read of G267 as declared. The arm (of six) is chosen on M3 dev onset windows of odd calendar days
(0-based day even), by the mean IoU minus base. Read: M3 dev even days (judge, April composition; all and the top coverage third) and
January / February 317 windows (F = all finite cells; month x panel x day clusters as G170's ratio). Records: every arm on both halves.
usage: TFB_M3VER=v3 py -B -X utf8 g268_intensity_read.py"""
import os, sys
os.environ.setdefault("TFB_M3VER", "v3")
import numpy as np, pandas as pd
sys.path.insert(0, r"C:/Users/user/tfb/src"); import m3v2
W = "C:/Users/user/tfb/work/t2/"; ARMS = [f"I{p}_{sg}" for p in (50, 60, 75) for sg in (60, 180)]
D = pd.read_csv(W + "g267_m3.csv"); assert len(D) == 100 and not m3v2.is_holdout(D.day.to_numpy()).any(); T = pd.read_csv(W + "g267_train.csv")
d_ = "C:/tfb_data/tfb_nn/nn_test_m3v3/"; R_ = np.load(d_ + "R.npy", mmap_mode="r"); o_ = np.load(d_ + "off.npy"); w_ = [str(x) for x in np.load(d_ + "window_id.npy", allow_pickle=True)]
COV = {x: float(np.asarray(R_[o_[i]:o_[i + 1], :12, 2], float).mean()) for i, x in enumerate(w_)}; D["cov"] = D.w.astype(str).map(COV)
odd = (D.day % 2 == 0).to_numpy(); E = D[~odd].reset_index(drop=True); top = E[E["cov"] > D["cov"].quantile(2 / 3)].reset_index(drop=True)
for a in ARMS:
    print(f"  {a}: odd days {(D[a] - D.base)[odd].mean():+.4f} | even days raw {(D[a] - D.base)[~odd].mean():+.4f} | Jan/Feb F {(T['F_' + a] - T.F_base).mean():+.4f} | windows changed (odd / even / Jan-Feb) {int((~np.isclose(D[a], D.base))[odd].sum())} / {int((~np.isclose(D[a], D.base))[~odd].sum())} / {int((~np.isclose(T['F_' + a], T.F_base)).sum())}")
best = max(ARMS, key=lambda a: (D[a] - D.base)[odd].mean()); print("chosen on odd days:", best, flush=True)
def ratio(Dt, nu, n_boot=2000):
    ids = pd.factorize(Dt.panel + "_" + Dt.month.astype(str) + "_" + Dt.day.astype(str))[0]; M = np.bincount(ids, nu); N = np.bincount(ids); Rm = np.random.default_rng(0).multinomial(len(M), np.full(len(M), 1 / len(M)), size=n_boot); return float(M.sum() / N.sum()), float(((Rm @ M) / (Rm @ N)).std())
for lab, g_ in (("M3 even days all", E), ("M3 even days top third", top)):
    s = m3v2.summary(g_, "base", best, "private", n_boot=2000, n_draw=500); print(f"{best} - base, {lab} ({len(g_)}): {s['t2_weighted']:+.4f} (SE {s['se']:.4f}, z {s['t2_weighted'] / max(s['se'], 1e-12):+.2f})", flush=True)
jm, js = ratio(T, (T["F_" + best] - T.F_base).to_numpy()); print(f"{best} - base, Jan/Feb F ({len(T)}): {jm:+.4f} (SE {js:.4f}, z {jm / max(js, 1e-12):+.2f})", flush=True)
