"""B20 (10/29, strategy upper-bound audit item 5): are the per-site jam durations stable from the training months to March?
B13's fill and tracking on the March release (validation: T1 masks and the T2 release blanks are missing data; events that
touch a release blank are dropped because their length is unknown). Known sites (b13_sites). Per site with >= 15 March events:
median duration (rows) in training vs March, KS distance and p; pooled: share of sites with KS p < 0.05.
Diagnostic only (March aggregates <= its own days; nothing is tuned)."""
import sys, numpy as np, pandas as pd
sys.path.insert(0, r"C:/Users/user/tfb/src"); sys.path.insert(0, r"C:/Users/user/tfb/audit")
from scipy.stats import ks_2samp
from cachelib import load
from common import WORK
from b13_jam_shapes import fill, runs
S = pd.read_parquet(WORK + "/b13_sites.parquet"); TR = pd.read_parquet(WORK + "/b13_events.parquet")
rows = []
for p, ss in S.groupby("panel"):
    P = load(p, "validation"); C = np.where(np.isnan(P.V), np.nan, (P.V <= P.vcut[None, :]).astype(float)); B = fill(C)
    blank = np.isnan(P.V).all(axis=1)
    active, done = {}, []
    for t in range(P.T):
        new = {}
        for a, b in runs(B[t]):
            prev = next((h for h in (b, b - 1, b + 1) if h in active and h not in [e["hid"] for e in new.values()]), None)
            ev = active.pop(prev) if prev is not None else {"t0": t, "heads": [], "hid": None}
            ev["heads"].append(b); ev["hid"] = b; new[b] = ev
        done += list(active.values()); active = new
    for ev in done:
        n = len(ev["heads"]); hm = int(np.bincount(ev["heads"]).argmax())
        if n < 12 or blank[max(0, ev["t0"] - 1):ev["t0"] + n + 1].any():
            continue
        if hm in set(ss["head"]):
            rows.append((p, hm, n))
    load.cache_clear()
M = pd.DataFrame(rows, columns=["panel", "head", "dur"]); out = []
for (p, h), g in M.groupby(["panel", "head"]):
    tr = TR[(TR.panel == p) & (TR["head"] == h) & (TR.dur >= 12)].dur
    if len(g) >= 15 and len(tr) >= 30:
        ks = ks_2samp(tr, g.dur); out.append((p, h, len(tr), len(g), tr.median(), g.dur.median(), ks.statistic, ks.pvalue))
O = pd.DataFrame(out, columns=["panel", "head", "n_train", "n_march", "med_train", "med_march", "ks", "p"])
print(O.round(3).to_string())
print(f"sites {len(O)} | KS p < 0.05: {(O.p < 0.05).mean():.2f} | median |med_march - med_train| {np.median(np.abs(O.med_march - O.med_train)):.1f} rows | median KS {O.ks.median():.3f}")

# selection check (10/29): drop the training events that touch March's release-blank rows placed at the same row of
# each training month (same month-relative row index) -> if the training medians fall to March's, the shift is selection
BL = {}
for p in S.panel.unique():
    P = load(p, "validation"); BL[p] = np.flatnonzero(np.isnan(P.V).all(axis=1)); load.cache_clear()
TR2 = TR[TR.dur >= 12].copy(); TR2["mstart"] = 0
for p in TR2.panel.unique():
    P = load(p, "train"); ym = np.array([t[:7] for t in P.times]); first = {m: int(np.flatnonzero(ym == m)[0]) for m in np.unique(ym)}
    sel = TR2.panel == p; TR2.loc[sel, "mstart"] = [first[d[:7]] for d in TR2.loc[sel, "day"]]; load.cache_clear()
keep = []
for e in TR2.itertuples():
    b = BL.get(e.panel, np.array([], int)); rel0, rel1 = e.t0 - e.mstart - 1, e.t0 - e.mstart + e.dur + 1
    keep.append(not ((b >= rel0) & (b <= rel1)).any())
TR2 = TR2[np.array(keep)]; out2 = []
for (p, h), g in M.groupby(["panel", "head"]):
    tr = TR2[(TR2.panel == p) & (TR2["head"] == h)].dur
    if len(g) >= 15 and len(tr) >= 30:
        ks = ks_2samp(tr, g.dur); out2.append((p, h, tr.median(), g.dur.median(), ks.statistic, ks.pvalue))
O2 = pd.DataFrame(out2, columns=["panel", "head", "med_train_sel", "med_march", "ks", "p"])
print(f"\nwith the same blank selection on the training months: sites {len(O2)} | KS p < 0.05: {(O2.p < 0.05).mean():.2f} | "
      f"median (med_march - med_train_sel) {np.median(O2.med_march - O2.med_train_sel):+.1f} rows | median KS {O2.ks.median():.3f}")

# masking check (10/29): the training months re-extracted with the release view (T1 target cells P.blank hidden, as in
# March) plus March's blank rows placed at the same month-relative rows
from common import PANELS
rows3 = []
for p in S.panel.unique():
    P = load(p, "train"); V = np.where(P.blank, np.nan, P.V); ym = np.array([t[:7] for t in P.times])
    for mth in np.unique(ym):
        f = np.flatnonzero(ym == mth); a0 = f[0]; b_ = BL[p][BL[p] < len(f)] + a0; V[b_] = np.nan
    C = np.where(np.isnan(V), np.nan, (V <= P.vcut[None, :]).astype(float)); B = fill(C); blank = np.isnan(V).all(axis=1)
    active, done = {}, []
    for t in range(P.T):
        new = {}
        for a, b in runs(B[t]):
            prev = next((h for h in (b, b - 1, b + 1) if h in active and h not in [e["hid"] for e in new.values()]), None)
            ev = active.pop(prev) if prev is not None else {"t0": t, "heads": [], "hid": None}
            ev["heads"].append(b); ev["hid"] = b; new[b] = ev
        done += list(active.values()); active = new
    hs = set(S[S.panel == p]["head"])
    for ev in done:
        n = len(ev["heads"]); hm = int(np.bincount(ev["heads"]).argmax())
        if n >= 12 and not blank[max(0, ev["t0"] - 1):ev["t0"] + n + 1].any() and hm in hs:
            rows3.append((p, hm, n))
    print(p, "nan share train-view %.3f" % np.isnan(V).mean(), flush=True); load.cache_clear()
T3 = pd.DataFrame(rows3, columns=["panel", "head", "dur"]); out3 = []
for (p, h), g in M.groupby(["panel", "head"]):
    tr = T3[(T3.panel == p) & (T3["head"] == h)].dur
    if len(g) >= 15 and len(tr) >= 30:
        ks = ks_2samp(tr, g.dur); out3.append((tr.median(), g.dur.median(), ks.statistic, ks.pvalue))
O3 = pd.DataFrame(out3, columns=["med_train_view", "med_march", "ks", "p"])
print(f"\nwith the release view on the training months: sites {len(O3)} | KS p < 0.05: {(O3.p < 0.05).mean():.2f} | "
      f"median (med_march - med_train_view) {np.median(O3.med_march - O3.med_train_view):+.1f} rows | median KS {O3.ks.median():.3f}")
for p in S.panel.unique()[:2]:
    Pv = load(p, "validation"); print(p, "nan share March %.3f" % np.isnan(Pv.V).mean()); load.cache_clear()
