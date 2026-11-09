"""Audit A14: t2_monthprior features under validation-like conditions.
Train months are scanned from each calendar month start with the organizer-selector replica (t2_selector.scan), as
the validation/private windows are.  For each onset window: mp_first / mp_tod computed (a) as in training (mpcache:
masked layer, no Task 2 gaps) and (b) as at inference (earlier windows of the same month leave rows T'+1..T'+18 blank).
Reported: value at the true onset links (label at T+30) and agreement of the two versions."""
import sys, numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, r"C:/Users/user/tfb/src")
from cachelib import load, T2_PANELS
from t2_windows import queue_matrix
from t2_selector import scan


def mp(qd, months, t, hide=None):
    D = qd.shape[0]; d = t // 288; m = months[d]
    prev = [i for i in range(d) if months[i] == m]
    if not prev:
        return None
    q = qd[prev].copy()
    if hide is not None:
        q &= ~hide[prev]
    first = np.zeros((len(prev), q.shape[2]), bool)
    for k in range(len(prev)):
        r = np.flatnonzero(q[k].any(axis=1))
        if len(r):
            first[k] = q[k, r[0]]
    tod = (t % 288) + 6; lo, hi = max(tod - 12, 0), min(tod + 13, 288)
    return first.mean(0), q[:, lo:hi].any(axis=1).mean(0)


rows = []
for p in T2_PANELS:
    P = load(p, "train"); Qm = queue_matrix(P)
    V = P.V.copy(); V[P.blank] = np.nan
    qmask = (np.nan_to_num(V, nan=1e9) <= P.vcut[None, :]) & P.E
    D = P.T // 288; qd = qmask[:D * 288].reshape(D, 288, P.n)
    months = np.array([s[:7] for s in P.day[:D]])
    starts = [int(np.flatnonzero(months == m)[0]) for m in np.unique(months)]
    for d0 in starts:
        on, og = scan(P, Qm, start=d0 * 288 + 12)
        allw = sorted(on + og)
        for t in on:
            if t // 288 >= D or months[t // 288] != months[d0]:
                continue
            hide = np.zeros((D, 288, P.n), bool)
            for t2 in allw:
                if t2 < t:
                    for r_ in range(t2 + 1, t2 + 19):
                        if r_ // 288 < D:
                            hide[r_ // 288, r_ % 288] = True
            a = mp(qd, months, t); b = mp(qd, months, t, hide)
            if a is None:
                rows.append(dict(panel=p, day=t // 288 - d0 + 1, no_prev=True)); continue
            lab = Qm[t + 6]
            fa, fb = a[0], b[0]
            rows.append(dict(panel=p, day=t // 288 - d0 + 1, no_prev=False, at_true_a=fa[lab].mean() if lab.any() else np.nan,
                             at_true_b=fb[lab].mean() if lab.any() else np.nan, argmax_hit_a=bool(lab[np.argmax(fa)]) if fa.max() > 0 else False,
                             argmax_hit_b=bool(lab[np.argmax(fb)]) if fb.max() > 0 else False, mae=np.abs(fa - fb).mean(), corr=np.corrcoef(fa, fb)[0, 1] if fa.std() > 0 and fb.std() > 0 else np.nan))
    load.cache_clear()
d = pd.DataFrame(rows)
print("onset windows (month-start scans):", len(d), " day-of-month distribution:", d.day.value_counts().sort_index().to_dict())
print("no earlier day in month (mp all NaN):", int(d.no_prev.sum()))
e = d[~d.no_prev]
print(f"mp_first at the true onset links: training view {e.at_true_a.mean():.3f}  inference view {e.at_true_b.mean():.3f}")
print(f"argmax(mp_first) is a true onset link: training view {e.argmax_hit_a.mean():.3f}  inference view {e.argmax_hit_b.mean():.3f}")
print(f"per-link agreement of the two views: mean |diff| {e.mae.mean():.3f}, corr {e["corr"].mean():.3f}")
