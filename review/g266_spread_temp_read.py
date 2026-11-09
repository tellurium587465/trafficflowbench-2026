"""G266 (11/04, ledger #225, declared by the strategy 21:20 before the results): the spread-dependent ongoing temperature, read clean.
(1) On M3 dev odd calendar days (0-based day even) only: the thirds' cut points of the 40 nets' spread (G263) and per third the best of
{1.0, 1.2, 1.4} on A's flow (G263's a12t10 / a12 / a12t14). (2) Even days: the rule against 1.2 throughout, all and the top coverage
third (judge, April composition). (3) January / February 316 windows (G265: capB0 four seeds' spread as the proxy, cut points re-placed
at the January / February thirds, the same per-third temperatures): A's and B's flows against 1.2, cluster bootstrap (month x panel x day).
B uses the same temperatures (as declared); B's own odd-day choice is printed as a record. usage: TFB_M3VER=v3 py -B -X utf8 g266_spread_temp_read.py"""
import os, sys
os.environ.setdefault("TFB_M3VER", "v3")
import numpy as np, pandas as pd
sys.path.insert(0, r"C:/Users/user/tfb/src"); import m3v2
W = "C:/Users/user/tfb/work/t2/"
A = pd.read_csv(W + "g263_dev.csv"); B = pd.read_csv(W + "g264_dev.csv")[["w", "b11", "b11t10", "b11t14"]]; D = A.merge(B, on="w"); assert len(D) == 356
d_ = "C:/tfb_data/tfb_nn/nn_test_m3v3/"; R_ = np.load(d_ + "R.npy", mmap_mode="r"); o_ = np.load(d_ + "off.npy"); w_ = [str(x) for x in np.load(d_ + "window_id.npy", allow_pickle=True)]
COV = {x: float(np.asarray(R_[o_[i]:o_[i + 1], :12, 2], float).mean()) for i, x in enumerate(w_)}; D["cov"] = D.w.astype(str).map(COV)
odd = (D.day % 2 == 0).to_numpy(); cut = np.quantile(D.spread[odd], [1 / 3, 2 / 3]); lay = lambda s, c: np.digitize(s, c)
D["lay"] = lay(D.spread.to_numpy(), cut)
CA = {1.0: "a12t10", 1.2: "a12", 1.4: "a12t14"}; CB = {1.0: "b11t10", 1.2: "b11", 1.4: "b11t14"}
TA = {l: max(CA, key=lambda te: D.loc[odd & (D.lay == l), CA[te]].mean()) for l in range(3)}; TB = {l: max(CB, key=lambda te: D.loc[odd & (D.lay == l), CB[te]].mean()) for l in range(3)}
print("M3 odd days: cut points", np.round(cut, 5), "| A temperatures by third", TA, "| B's own (record)", TB, flush=True)
D["a_rule"] = [D.at[i, CA[TA[l]]] for i, l in zip(D.index, D.lay)]; D["b_rule"] = [D.at[i, CB[TA[l]]] for i, l in zip(D.index, D.lay)]; D["b_own"] = [D.at[i, CB[TB[l]]] for i, l in zip(D.index, D.lay)]
E = D[~odd].reset_index(drop=True); top = E[E["cov"] > D["cov"].quantile(2 / 3)].reset_index(drop=True)
for a_, b_, nm in (("a12", "a_rule", "A"), ("b11", "b_rule", "B (A's temperatures)"), ("b11", "b_own", "B (own, record)")):
    for lab, g_ in (("even days all", E), ("even days top third", top)):
        s = m3v2.summary(g_, a_, b_, "private", n_boot=2000, n_draw=500); print(f"M3 {nm}, {lab} ({len(g_)}): {s['t2_weighted']:+.4f} (SE {s['se']:.4f}, z {s['t2_weighted'] / max(s['se'], 1e-12):+.2f})", flush=True)
J = pd.read_csv(W + "g265_jf.csv"); cj = np.quantile(J.spread, [1 / 3, 2 / 3]); J["lay"] = lay(J.spread.to_numpy(), cj); cl = (J.month.astype(str) + "_" + J.panel + "_" + J.day.astype(str)).to_numpy()
def boot(d, cl, n=2000):
    ids = pd.factorize(cl)[0]; M_ = np.bincount(ids, d); N_ = np.bincount(ids); Rm = np.random.default_rng(0).multinomial(len(M_), np.full(len(M_), 1 / len(M_)), size=n); return float(((Rm @ M_) / (Rm @ N_)).std())
for fl, tt, nm in (("A", TA, "A"), ("B", TA, "B (A's temperatures)"), ("B", TB, "B (own, record)")):
    ch = np.array([J[f"{fl}{tt[l]}"].iloc[i] for i, l in enumerate(J.lay)]); d = ch - J[f"{fl}1.2"].to_numpy(); se = boot(d, cl)
    print(f"Jan/Feb {nm} ({len(J)}, cuts {np.round(cj, 5)}): {d.mean():+.4f} (SE {se:.4f}, z {d.mean() / max(se, 1e-12):+.2f}) | windows changed {int((np.abs(d) > 1e-12).sum())}", flush=True)
