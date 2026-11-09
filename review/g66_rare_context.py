"""G66 (10/22 night, premise of (i)/(iii) of the rare-site onset reopen): can the window CONTEXT tell that this onset will be
at a new site? Train months, selector-like onset windows (as G54). Leave-month-out rarity: f_lomo(link) = share of the
OTHER months' selector-like onset windows of the same panel x weekend x 2-hour slot in which the link is queued at T+30;
"new-site onset" = every true T+30 link has f_lomo <= 0.1. Context per window (all at or before T):
  hour, weekend, corridor queued earlier today (q_today > 0 anywhere), corridor demand level (dem_level, mean),
  same-day other panels: share of observed queued cells in T-12..T minus that panel's typical share at the same
  weekend x slot (train months), averaged over the other panels (cross-panel 'abnormal day').
Prints the rate of new-site onsets and the AUC of each context variable for new vs usual site, and a small logistic
model's leave-one-month-out AUC.
usage: python g66_rare_context.py"""
import sys, os
os.environ["TFB_T2CACHE"] = "2_noT_af"                      # windows in the order of nn_data_afdem (checked: same off)
sys.path.insert(0, r"C:/Users/user/tfb/src")
sys.argv = [sys.argv[0], "0"]
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from t2_eval_oof import T
from cachelib import load, T2_PANELS
from common import NN
XN = np.load(f"{NN}/nn_data_afdem/X.npy", mmap_mode="r"); OFFN = np.load(f"{NN}/nn_data_afdem/off.npy")
NAMES = [str(x) for x in np.load(f"{NN}/nn_data_afdem/names.npy", allow_pickle=True)]
assert len(OFFN) - 1 == T.NWIN

SELON = [k for k in range(T.NWIN) if T.META["is_sel"][k] and str(T.META["cond"][k]) == "queue_onset"]
mon = lambda k: str(T.META["date"][k])[:7]
we = lambda k: pd.Timestamp(str(T.META["date"][k])[:10]).dayofweek >= 5
key = lambda k: (str(T.META["panel"][k]), we(k), (int(T.META["t"][k]) % 288) // 24)
CNT, DEN = {}, {}
for k in SELON:
    lab, el = T.LAB[k]; y = (lab[5] & el[5]).astype(float); kk = key(k) + (mon(k),)
    CNT[kk] = CNT.get(kk, 0) + y; DEN[kk] = DEN.get(kk, 0) + 1
months = sorted({mon(k) for k in SELON})

# per panel: share of observed queued cells per row, and its typical value per weekend x slot
SHARE, TYP = {}, {}
for p in T2_PANELS:
    P = load(p, "train"); fin = np.isfinite(P.V); q = fin & (P.V <= P.vcut[None, :])
    sh = q.sum(1) / np.maximum(fin.sum(1), 1); SHARE[p] = sh
    wk = np.array([pd.Timestamp(s[:10]).dayofweek >= 5 for s in P.times]); sl = np.arange(len(sh)) % 288 // 12
    TYP[p] = {(w, s): float(sh[(wk == w) & (sl == s)].mean()) for w in (False, True) for s in range(24)}
    TYP[p]["wk"], TYP[p]["sl"], TYP[p]["times"] = wk, sl, P.times; TYP[p]["pos"] = {s_: i for i, s_ in enumerate(P.times)}
IQ = NAMES.index("q_today"); IDL = NAMES.index("dem_level")
rows = []
for k in SELON:
    lab, el = T.LAB[k]; y = lab[5] & el[5]; tl = np.flatnonzero(y)
    if not len(tl):
        continue
    num = sum(CNT[key(k) + (m,)] for m in months if m != mon(k) and key(k) + (m,) in CNT)
    den = sum(DEN[key(k) + (m,)] for m in months if m != mon(k) and key(k) + (m,) in DEN)
    f = num / den if den else np.zeros(len(y))
    X = np.asarray(XN[OFFN[k]:OFFN[k + 1]]); p0 = str(T.META["panel"][k]); t = int(T.META["t"][k])
    ts = TYP[p0]["times"][t]; an = []
    for p in T2_PANELS:
        if p == p0:
            continue
        j = TYP[p]["pos"].get(ts)                           # the same time stamp in the other panel
        if j is None:
            continue
        an.append(float(np.mean(SHARE[p][max(0, j - 12):j + 1])) - TYP[p][(bool(TYP[p]["wk"][j]), int(TYP[p]["sl"][j]))])
    rows.append(dict(k=k, month=mon(k), new=bool((f[tl] <= 0.1).all()), hour=(t % 288) // 12, weekend=we(k),
                     today=bool((np.nan_to_num(X[:, IQ]) > 0).any()), dem=float(np.nanmean(X[:, IDL])) if IDL is not None else np.nan,
                     other_anom=float(np.mean(an)) if an else np.nan))
R = pd.DataFrame(rows); R.to_csv("C:/Users/user/tfb/work/t2/g66_rare_context.csv", index=False)
print(f"train-month selector-like onset windows {len(R)}, new-site onsets {R.new.sum()} ({R.new.mean():.2f})")
F = [c for c in ("hour", "weekend", "today", "dem", "other_anom") if R[c].notna().any()]
for c in F:
    ok = R[c].notna()
    print(f"  {c}: AUC new vs usual {roc_auc_score(R.new[ok], R[c][ok].astype(float)):.3f} | mean new {R[c][R.new].astype(float).mean():.3f} / usual {R[c][~R.new].astype(float).mean():.3f}")
R2 = R.dropna(subset=F).copy(); R2["h2"] = (R2.hour - 12) ** 2; F2 = F + ["h2"]
pr = np.zeros(len(R2))
for m in months:
    tr, te = R2.month != m, R2.month == m
    if te.sum() and R2.new[tr].nunique() == 2:
        pr[te.values] = LogisticRegression(max_iter=2000).fit(R2.loc[tr, F2].astype(float), R2.new[tr]).predict_proba(R2.loc[te, F2].astype(float))[:, 1]
print(f"logistic (context only) leave-one-month-out AUC {roc_auc_score(R2.new, pr):.3f}")
