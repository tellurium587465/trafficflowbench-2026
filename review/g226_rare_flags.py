"""G226 (11/03, ledger #193): which training onset windows are rare births, for the loss weight RW of t2_nn3r.py.
Pool = nn_data_afdem. Months = fold 2's training months (2030-06 .. 2030-11; December, January and February are the OOF
months and are not read). Rarity = rarity_table's definition (t2_phys_onset: per panel x weekend x 2-hour slot, the share
of selector onset windows with the link queued at T+30, observed eligible cells), computed leave-one-month-out within those
months; a window is a rare birth when every link queued at T+30 has frequency <= 0.1 (G76) and the birth lasts >= 30 min
(some birth link queued, or unobserved, in each of rows T+30 .. T+55 of the panel's queue matrix).
Writes C:/tfb_data/tfb_nn/rare193/rare.npy (bool per pool window) and prints the share and the RW for a 17.5 % share.
usage: py -B -X utf8 g226_rare_flags.py"""
import os, sys, numpy as np, pandas as pd
os.environ["TFB_T2_NOT"] = "1"; os.environ["TFB_T2_ALLFIN"] = "1"
sys.path.insert(0, r"C:/Users/user/tfb/src")
from cachelib import load
from t2_windows import queue_matrix
from common import NN

d = f"{NN}/nn_data_afdem/"; A = {f: np.load(d + f + ".npy", allow_pickle=True) for f in ("panel", "t", "cond", "date", "is_sel", "off")}
Y = np.load(d + "Y.npy", mmap_mode="r"); EL = np.load(d + "EL.npy", mmap_mode="r"); off = A["off"]
mon = np.array([str(x)[:7] for x in A["date"]]); TRM = ("2030-06", "2030-07", "2030-08", "2030-09", "2030-10", "2030-11")
key = lambda k: (str(A["panel"][k]), pd.Timestamp(str(A["date"][k])[:10]).dayofweek >= 5, (int(A["t"][k]) % 288) // 24)
q30 = lambda k: (np.asarray(Y[off[k]:off[k + 1], 5]) > 0) & (np.asarray(EL[off[k]:off[k + 1], 5]) > 0)
cnt, den = {}, {}
for k in np.flatnonzero(A["is_sel"] & (A["cond"] == "queue_onset") & np.isin(mon, TRM)):
    kk = key(k); m = mon[k]; cnt.setdefault(kk, {}).setdefault(m, 0); den.setdefault(kk, {}).setdefault(m, 0)
    cnt[kk][m] = cnt[kk][m] + q30(k).astype(float); den[kk][m] += 1
rare = np.zeros(len(A["t"]), bool); site = np.zeros(len(A["t"]), bool); n_on = 0; chk = []
ON = np.flatnonzero((A["cond"] == "queue_onset") & np.isin(mon, TRM))
for p in sorted(set(A["panel"][ON])):
    P = load(p, "train"); Qm = queue_matrix(P); obs = np.isfinite(P.V) & P.E
    for k in ON[A["panel"][ON] == p]:
        kk = key(k); m = mon[k]; t = int(A["t"][k]); q = q30(k); n_on += 1
        if len(chk) < 200:
            chk.append(bool(np.array_equal(q, Qm[t + 6] & obs[t + 6])))
        c = sum((v for mm, v in cnt.get(kk, {}).items() if mm != m), np.zeros(P.n)); n = sum(v for mm, v in den.get(kk, {}).items() if mm != m)
        f = c / n if n else np.zeros(P.n)
        site[k] = q.any() and bool((f[q] <= 0.1).all())
        rare[k] = site[k] and bool(((Qm | ~obs)[t + 6:t + 12][:, q].any(axis=1)).all())
    load.cache_clear()
s = rare[ON].mean(); rw = 0.175 * (1 - s) / (s * (1 - 0.175)) if s > 0 else float("nan")
print(f"alignment: T+30 labels == queue matrix row t+6 on {sum(chk)} of {len(chk)} windows")
print(f"fold-2 training onset windows {n_on}: rare site {int(site[ON].sum())} ({site[ON].mean():.3f}), rare birth >= 30 min {int(rare[ON].sum())} ({s:.3f}); RW for a 17.5 % share {rw:.2f}")
os.makedirs(f"{NN}/rare193", exist_ok=True); np.save(f"{NN}/rare193/rare.npy", rare)
