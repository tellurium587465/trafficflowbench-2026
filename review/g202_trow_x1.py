"""G202 (11/02, ledger #164c): the row-T gain inside the full flow. Plain X1 (G141 arm pl, days before D, the path of t2_m3v2_x1) on capB0 and on capBT, seeds 0-3, fold 2 weights: the capBT runs read v_T and q_T from the March pool and from M3 (nn_data_marcht, nn_test_m3v3t), the capB0 runs the original inputs. A6's flow (0.8 (0.8 X1 + 0.2 tfm4) + 0.2 LightGBM -> cone -> seen x0.85 -> best_set -> far gate), judge on M3 dev. G142 and its csv are not changed.
  data  : builds nn_data_marcht (X with row T filled from the release, the rest copied)
  score : the read, after G141's fits
usage: py -B -X utf8 g202_trow_x1.py data | TFB_M3VER=v3 py -B -X utf8 g202_trow_x1.py score"""
import os, sys, pickle, shutil
sys.dont_write_bytecode = True
MODE = sys.argv[1]
if MODE == "data":
    import numpy as np, warnings
    os.environ["TFB_T2_NOT"] = "1"; os.environ["TFB_T2_ALLFIN"] = "1"; warnings.filterwarnings("ignore"); sys.path.insert(0, r"C:/Users/user/tfb/src")
    from cachelib import load
    from t2_model import PanelInputs
    NN = "C:/tfb_data/tfb_nn/"; d = NN + "nn_data_march/"; O = NN + "nn_data_marcht/"; os.makedirs(O, exist_ok=True)
    for f in os.listdir(d):
        if f != "X.npy" and not os.path.exists(O + f):
            shutil.copy(d + f, O + f)
    X = np.array(np.load(d + "X.npy", mmap_mode="r")); names = [str(x) for x in np.load(d + "names.npy", allow_pickle=True)]; off = np.load(d + "off.npy"); pan = [str(x) for x in np.load(d + "panel.npy", allow_pickle=True)]; t_ = np.load(d + "t.npy"); iv, iq = names.index("v_T"), names.index("q_T"); assert np.isnan(X[:, iv]).all(); C = {}
    for k in range(len(off) - 1):
        p = pan[k]
        if p not in C:
            P = load(p, "validation"); PI = PanelInputs(P); C[p] = (PI.mask_v / P.vf[None, :], PI.mask_q / P.cap[None, :])
        a, b = int(off[k]), int(off[k + 1]); assert b - a == C[p][0].shape[1]; X[a:b, iv] = C[p][0][int(t_[k])]; X[a:b, iq] = C[p][1][int(t_[k])]
    np.save(O + "X.npy", X); print("windows", len(off) - 1, "share of row T seen", round(float(np.isfinite(X[:, iv]).mean()), 4)); sys.exit()
os.environ.setdefault("TFB_M3VER", "v3")
s = open(r"C:/Users/user/tfb/review/g142_x1_arms_score.py", encoding="utf-8").read()


def sub(a, b):
    global s
    assert s.count(a) == 1, a[:60]
    s = s.replace(a, b)


sub('TAGS = [f"c24f_attnglob_s{s}" for s in (0, 1)]', 'TAGS = []')
sub('def arm(name):\n    PR = []\n    for t in TAGS:', 'def arm(name, TAGS):\n    PR = []\n    for t in TAGS:')
sub('return {key: (PR[0][key] + PR[1][key]) / 2 for key in PR[0]}', 'return {key: sum(d_[key] for d_ in PR) / len(PR) for key in PR[0]}')
sub('AR = {n: arm(n) for n in ("sd", "hl3", "hl7", "pl")}; AR = {n: a for n, a in AR.items() if a}; TT = arm("tta");', '''def fill0(tag_, dd_):                                           
    return {(w_, k_): v_ for (w_, k_, s_), v_ in dd_.items() if s_ == 1.0}     # the fit's key carries the flow scale


X1 = fill0("capB0", arm("pl", [f"capB0_s{i}_fold2030-12" for i in range(4)])); AR = {"BT": fill0("capBT", arm("pl", [f"capBT_s{i}_fold2030-12" for i in range(4)]))}; TT = None;''')
sub('TN = TN if len(TN) == len(ON) else {}', 'TN = {}')
sub('D.to_csv(OUT + "g142.csv", index=False)', 'D.to_csv(OUT + "g202.csv", index=False)')
sys.argv = [sys.argv[0]]; exec(compile(s, "g142c", "exec"))
