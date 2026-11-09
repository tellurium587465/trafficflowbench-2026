"""G97 (10/23, pre-declared STATE 10/23 21:49): can X1 be rescued on ONSET? M3 v3 onset windows, e4s_a seeds 0, 1 (train months
only, attnglob; the mean of the two), the k expectation with the selector-simulated blanks, the evidence rule and decoder of
the M3 v2 gate. Arms: base (saved models) / x1 (plain X1, every k) / x1k4 (plain X1 only for k >= 4, the saved model
below) / l2sp0.1, l2sp1, l2sp10 (L2-SP fine-tunes for k >= 4, t2_m3v2_x1.py X1_KMIN=4 X1_L2SP=lambda).
Writes work/t2/g97_x1_onset<ver>.csv; judge with src/judge.py --cond queue_onset --k 4.
usage: TFB_M3VER=v3 python g97_x1_onset_rescue.py"""
import os, sys, pickle
os.environ.setdefault("TFB_M3VER", "v3")
sys.path.insert(0, r"C:/Users/user/tfb/src"); sys.argv = [sys.argv[0], "0,1,2,3"]
import t2_m3_gate as G, t2_m3v2_gate as V, m3v2

W = G.W; VER = G.VER; PRE = G.PRE
ld = lambda f: pickle.load(open(W + f, "rb")); av = lambda a, b: {k: (a[k] + b[k]) / 2 for k in a}
BASE = av(ld(f"probs_{PRE}__e4s_a_s0.pkl"), ld(f"probs_{PRE}__e4s_a_s1.pkl"))
ARM = {"x1": av(ld(f"probs_m3{VER}v2x1__e4s_a_s0.pkl"), ld(f"probs_m3{VER}v2x1__e4s_a_s1.pkl"))}
for lam in ("0.1", "1", "10"):
    ARM[f"l2sp{lam}"] = av(ld(f"probs_m3{VER}v2x1l2sp{lam}__e4s_a_s0.pkl"), ld(f"probs_m3{VER}v2x1l2sp{lam}__e4s_a_s1.pkl"))
cols = {c: {} for c in ("base", "x1", "x1k4", "l2sp0.1", "l2sp1", "l2sp10")}
for w in G.WID:
    m = G.META[w]
    if m["cond"] != "queue_onset":
        continue
    kd = m3v2.kdist(m["day"]); sc = lambda p, k: G.iou(V.onset_pred(p, m, "cur", k), m)
    cols["base"][w] = sum(pr * sc(BASE[w], k) for k, pr in kd.items())
    cols["x1"][w] = sum(pr * sc(ARM["x1"][(w, k)], k) for k, pr in kd.items())
    cols["x1k4"][w] = sum(pr * sc(ARM["x1"][(w, k)] if k >= 4 else BASE[w], k) for k, pr in kd.items())
    for lam in ("0.1", "1", "10"):
        cols[f"l2sp{lam}"][w] = sum(pr * sc(ARM[f"l2sp{lam}"][(w, k)], k) for k, pr in kd.items())
D = V.frame(cols).dropna(subset=list(cols)); D.to_csv(W + f"g97_x1_onset{VER}.csv", index=False); print("wrote", len(D), "onset windows")
