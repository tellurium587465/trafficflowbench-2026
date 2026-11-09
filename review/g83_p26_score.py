"""G83 (INSTRUMENT_V3_SPEC §2.6, 10/23): score the synthetic proxy of "a shifted month in training helps on other shifted
months" (work/p26_chain_1023.sh). Arm A = train months + C0-C5 ongoing; arm B = A + the whole world C6. Test sets never in
training: the windows of C7, C8, C9 (both conditions; NN/nn_test_c24w789, built here from nn_data_twinC24) and the M3
dev windows (real March, hold-out excluded). Seeds 0,1 averaged per arm. Synthetic: IoU of the expected-IoU set (onset
at T+30, ongoing all rows, eligible cells), paired B - A per window, SE clustered by panel x world. M3: t2_m3v2_gate.score
and m3v2.summary (model A/B -> z also / 1.25).
usage: python g83_p26_score.py build | score"""
import os, sys, pickle, subprocess
sys.path.insert(0, r"C:/Users/user/tfb/src")
step = sys.argv[1]
import numpy as np, pandas as pd
from cachelib import load
from common import NN

W = "C:/Users/user/tfb/work/t2/"; SRC = "C:/Users/user/tfb/src"; S = f"{NN}/nn_data_twinC24"; D = f"{NN}/nn_test_c24w789"
MONTHS = [load("D7_I10_E", f"twinC{k}").times[0][:7] for k in (7, 8, 9)]
date = np.load(f"{S}/date.npy", allow_pickle=True); cond = np.load(f"{S}/cond.npy", allow_pickle=True); off = np.load(f"{S}/off.npy")
panel = np.load(f"{S}/panel.npy", allow_pickle=True)
ks = np.flatnonzero(np.isin([str(x)[:7] for x in date], MONTHS))
if step == "build":
    X = np.load(f"{S}/X.npy", mmap_mode="r"); os.makedirs(D, exist_ok=True)
    np.save(f"{D}/X.npy", np.vstack([np.asarray(X[off[k]:off[k + 1]]) for k in ks]))
    np.save(f"{D}/off.npy", np.concatenate([[0], np.cumsum([off[k + 1] - off[k] for k in ks])]))
    np.save(f"{D}/window_id.npy", np.array([f"C24_{k}" for k in ks])); np.save(f"{D}/cond.npy", cond[ks])
    print("p26 test set C7-C9:", len(ks), "windows", flush=True); sys.exit()

from t2_model import best_set
TAGS = [f"p26{a}_s{s}" for a in "AB" for s in (0, 1)]
for td, out in (("nn_test_c24w789", "c24w789"), ("nn_test_m3", "m3")):
    todo = [t for t in TAGS if not os.path.exists(W + f"probs_{out}__{t}.pkl")]
    if todo:
        subprocess.run([sys.executable, "t2_nn_cpu_infer.py", ",".join(todo), td, out], cwd=SRC, check=True)
ld = lambda o, t: pickle.load(open(W + f"probs_{o}__{t}.pkl", "rb"))
arm = lambda o, a: (lambda x, y: {w: (x[w] + y[w]) / 2 for w in x})(ld(o, f"p26{a}_s0"), ld(o, f"p26{a}_s1"))
PA, PB = arm("c24w789", "A"), arm("c24w789", "B")
Y = np.load(f"{S}/Y.npy", mmap_mode="r"); E = np.load(f"{S}/EL.npy", mmap_mode="r"); rows = []
for k in ks:
    w = f"C24_{k}"; y = np.asarray(Y[off[k]:off[k + 1]]).T.astype(bool); e = np.asarray(E[off[k]:off[k + 1]]).T.astype(bool); r = dict(k=k, cond=str(cond[k]), cl=f"{panel[k]}|{str(date[k])[:7]}")
    for name, P in (("A", PA), ("B", PB)):
        p = P[w]
        if cond[k] == "queue_onset":
            s_ = best_set(p[5], e[5]); yy, ee = y[5], e[5]
        else:
            s_ = best_set(p.ravel(), e.ravel()).reshape(p.shape); yy, ee = y, e
        u = ((s_ | yy) & ee).sum(); r[name] = 1.0 if u == 0 else float((s_ & yy & ee).sum() / u)
    rows.append(r)
R = pd.DataFrame(rows); R.to_csv(W + "g83_p26.csv", index=False)
for c in ("queue_onset", "queue_ongoing"):
    g = R[R.cond == c]; d = g.B - g.A; cm = d.groupby(g.cl).sum(); nc = g.cl.value_counts()
    se = np.sqrt(len(cm) / (len(cm) - 1) * ((cm - cm.index.map(nc) * d.mean()) ** 2).sum()) / len(d)
    print(f"synthetic C7-C9 {c}: n {len(g)}, A {g.A.mean():.4f}, B {g.B.mean():.4f}, B - A {d.mean():+.4f} ({d.mean() / se:+.1f} SE)")
sys.argv = [sys.argv[0], "0"]
import t2_m3_gate as G, t2_m3v2_gate as V, m3v2
MA, MB = arm("m3", "A"), arm("m3", "B")
Dm = V.frame({"A": {w: V.score(MA[w], G.META[w]) for w in G.WID}, "B": {w: V.score(MB[w], G.META[w]) for w in G.WID}})
s = m3v2.summary(Dm[~m3v2.is_holdout(Dm.day)], "A", "B", "private")
print(m3v2.fmt("M3 dev B - A", s), f"| z x1.25 {s['t2_weighted'] / (1.25 * s['se']):+.2f}")
