"""Do the validation/private months activate bottlenecks the train months never did? (no labels needed)

Per onset window: link of maximum arriving-demand / downstream-capacity ratio (arr30/dnmin) and the model's
predicted head at T+30. Compare, train(selector-like) vs validation vs private:
  - share of windows whose hazard argmax is within 3 links of a train-recurrent onset site (top-3 per panel)
  - share whose predicted head is at a recurrent site, and agreement between prediction and hazard argmax.
"""
import pickle, numpy as np
from collections import Counter
from t2_eval_oof import T

D = {k: np.load(f"E:/tfb_nn/nn_data/{k}.npy", allow_pickle=True) for k in ("X", "off")}
Dt = {k: np.load(f"E:/tfb_nn/nn_test/{k}.npy", allow_pickle=True) for k in ("X", "off", "window_id", "cond")}
IA = 73                                                   # arr6 = arr30_over_dnmin
LEN2P = {}
heads = {}
for k in range(T.NWIN):
    if T.META["cond"][k] == "queue_onset":
        y = T.LAB[k][0][-1].astype(bool)
        if y.any():
            heads[k] = int(np.flatnonzero(y).max())
for k in heads:
    LEN2P[D["off"][k + 1] - D["off"][k]] = T.META["panel"][k]
top = {p: [l for l, _ in Counter(h for k, h in heads.items() if T.META["panel"][k] == p).most_common(3)] for p in set(LEN2P.values())}
near = lambda p, i: min(abs(i - t) for t in top[p]) <= 3

oof = pickle.load(open(r"C:/Users/user/tfb/work/t2/oof_l127_noT_arr.pkl", "rb"))
probs = pickle.load(open(r"C:/Users/user/tfb/work/t2/probs_v10.pkl", "rb"))
rows = {"train_sel": [], "validation": [], "private": []}
for k in heads:
    if not T.META["is_sel"][k] or k not in oof:
        continue
    a, b = D["off"][k], D["off"][k + 1]; p = T.META["panel"][k]
    h = np.nan_to_num(D["X"][a:b, IA], nan=-1); ph = int(np.argmax(oof[k][-1]))
    rows["train_sel"].append((near(p, int(np.argmax(h))), near(p, ph), abs(ph - int(np.argmax(h))) <= 3, near(p, heads[k])))
for i, w in enumerate(Dt["window_id"]):
    if Dt["cond"][i] != "queue_onset":
        continue
    a, b = Dt["off"][i], Dt["off"][i + 1]; p = LEN2P[b - a]
    split = "validation" if "valid" in str(w) else "private"
    h = np.nan_to_num(Dt["X"][a:b, IA], nan=-1); pr = probs[str(w)]; ph = int(np.argmax(pr[-1]))
    rows[split].append((near(p, int(np.argmax(h))), near(p, ph), abs(ph - int(np.argmax(h))) <= 3, np.nan))
print("example window id:", Dt["window_id"][0])
for s, r in rows.items():
    r = np.array(r, float)
    print(f"{s:10s} n={len(r):4d}  hazard-argmax at recurrent site {np.nanmean(r[:,0]):.3f} | predicted head at recurrent site {np.nanmean(r[:,1]):.3f}"
          f" | pred==hazard(+-3) {np.nanmean(r[:,2]):.3f} | true head at recurrent site {np.nanmean(r[:,3]) if s=='train_sel' else float('nan'):.3f}")
