"""G333 (11/05, ledger #264b, measurement only): does the box vanish by T+30, per (window, box)? Boxes = the last-queue runs (lq), head = the
downstream end. Target: no true queued cell on the box's links at T+30 among its scored cells (boxes without a scored T+30 cell left out).
(a) the network: 1 - mean p at T+30 over the box's links (k-weighted; M3 = A17's mix of G284, January / February = capB0 of G285, before
the temperature); (b) the month's earlier days (before the window's day): the days with the head (+-1 link) queued at the window's clock
T-5, the share of them where the head (+-1) is no longer queued at clock T+30 (visible rows only; NaN when fewer than 2 days); (c, record)
the training months' vanish rate of a head queued at a row, by head link x 2-hour slot x age (rows queued: 1-3 / 4-12 / 13+), with a
pseudo-count of 1 vanish in 2. AUCs of (a), (a)+(b), (a)+(c): logistic on logit(a), b (mean-filled + missing flag) / logit(c), fitted on
January / February and read on M3, and the reverse. Reading (#264b): M3 AUC of (a)+(b) >= (a) + 0.02. usage: TFB_M3VER=v3 py -B -X utf8 g333_box_vanish.py"""
import os, sys, pickle, importlib.util
os.environ.setdefault("TFB_M3VER", "v3")
import numpy as np, pandas as pd
R0 = r"C:/Users/user/tfb/"; sys.path.insert(0, r"C:/Users/user/tfb_a8/src"); sys.path.insert(0, R0 + "src")
from cachelib import load, T2_PANELS
import t2_m3_gate as G
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
W = R0 + "work/t2/"; lg = lambda p: np.log(np.clip(p, 1e-4, 1 - 1e-4) / (1 - np.clip(p, 1e-4, 1 - 1e-4)))


def runs(m):
    out, j = [], 0
    while j < len(m):
        if m[j]:
            e_ = j
            while e_ + 1 < len(m) and m[e_ + 1]:
                e_ += 1
            out.append((j, e_)); j = e_ + 1
        else:
            j += 1
    return out


def qmat(V, vcut):
    return np.isfinite(V), np.isfinite(V) & (V <= vcut[None, :])


def prev_share(O, Qm, day, tod5, tod30, hd, n):
    lo, hi = max(hd - 1, 0), min(hd + 2, n); cnt = van = 0
    for d in range(day):
        r5, r30 = d * 288 + tod5, d * 288 + tod30
        if r30 >= len(Qm) or not Qm[r5, lo:hi].any():
            continue
        if not O[r30, lo:hi].any():
            continue
        cnt += 1; van += not Qm[r30, lo:hi].any()
    return van / cnt if cnt >= 2 else np.nan


TAB = {}
for p in T2_PANELS:
    P = load(p, "train"); O, Qm = qmat(P.V, P.vcut); n = P.n; hd = Qm & ~np.concatenate([Qm[:, 1:], np.zeros((len(Qm), 1), bool)], axis=1)
    age = np.zeros_like(Qm, dtype=int)
    for r in range(1, len(Qm)):
        age[r] = np.where(Qm[r], age[r - 1] + 1, 0)
    num = np.ones((n, 12, 3)); den = np.full((n, 12, 3), 2.0); rr, jj = np.nonzero(hd[:-7]); fut = O[rr + 7, jj]; rr, jj = rr[fut], jj[fut]
    van = ~Qm[rr + 7, jj]; sl = (rr % 288) // 24; ab = np.digitize(age[rr, jj], [4, 13])
    np.add.at(num, (jj, sl, ab), van); np.add.at(den, (jj, sl, ab), 1.0); TAB[p] = num / den; load.cache_clear()
print("training vanish table built", flush=True)


def tab_c(p, hd, t, a):
    return TAB[p][hd, (t % 288) // 24, int(np.digitize(a, [4, 13]))]


rows = []
VM = {}
for p in T2_PANELS:
    P = load(p, "validation"); VM[p] = qmat(P.V, P.vcut); load.cache_clear()
D4 = pickle.load(open(W + "g284_og_pretemp_dev.pkl", "rb"))
for w, v in D4.items():
    m = G.META[w]; i_ = G.POS[w]; lq = np.asarray(G.X[G.OFF[i_]:G.OFF[i_ + 1], G.CI], float) == 1; y = m["y"].T.astype(bool); e = m["m"].T.astype(bool); t = m["t"]; O, Qm = VM[m["panel"]]
    for a, b in runs(lq):
        if not e[-1, a:b + 1].any():
            continue
        pa = sum(pr * float(1 - p[-1, a:b + 1].mean()) for pr, p, *_ in v["k"].values()); age = 0
        for r in range(t - 1, max(t - 13, 0), -1):
            if Qm[r, a:b + 1].any():
                age += 1
            else:
                break
        rows.append(dict(src="M3", vanish=not (y[-1, a:b + 1] & e[-1, a:b + 1]).any(), a=pa, b=prev_share(O, Qm, m["day"], (t - 1) % 288, (t + 6) % 288, b, len(lq)), c=tab_c(m["panel"], b, t, age)))
spec = importlib.util.spec_from_file_location("g119", R0 + "review/g119_month_postproc.py"); h = importlib.util.module_from_spec(spec); sys.argv = ["g119", "run"]; spec.loader.exec_module(h)
D5 = pickle.load(open(W + "g285_jf_pretemp.pkl", "rb")); T_ = h.T
for p in T2_PANELS:
    P = load(p, "train"); Ot, Qt = qmat(np.where(P.blank, np.nan, P.V), P.vcut); mon = np.array([s[:7] for s in P.times])
    for k, v in [(k, v) for k, v in D5.items() if v["panel"] == p]:
        t = int(T_.META["t"][k]); lq = np.asarray(h.XN[h.OFFN[k]:h.OFFN[k + 1]][:, h.G.CI], float) == 1; y, e = v["y"], v["e"]; r0 = int(np.flatnonzero(mon == v["month"])[0])
        for a, b in runs(lq):
            if not e[-1, a:b + 1].any():
                continue
            pa = sum(pr * float(1 - pb[-1, a:b + 1].mean()) for pr, pb in v["k"].values()); age = 0
            for r in range(t - 1, max(t - 13, 0), -1):
                if Qt[r, a:b + 1].any():
                    age += 1
                else:
                    break
            rows.append(dict(src="JF", vanish=not (y[-1, a:b + 1] & e[-1, a:b + 1]).any(), a=pa, b=prev_share(Ot[r0:], Qt[r0:], v["day"], (t - 1) % 288, (t + 6) % 288, b, len(lq)), c=tab_c(p, b, t, age)))
    load.cache_clear()
D = pd.DataFrame(rows); D.to_csv(W + "g333_boxes.csv", index=False)
for s_ in ("M3", "JF"):
    d = D[D.src == s_]; print(f"{s_}: boxes {len(d)}, vanish rate {d.vanish.mean():.3f}, (b) available {d.b.notna().mean():.3f}, AUC (a) {roc_auc_score(d.vanish, d.a):.3f}, (b) alone on available {roc_auc_score(d.vanish[d.b.notna()], d.b[d.b.notna()]):.3f}, (c) alone {roc_auc_score(d.vanish, d.c):.3f}", flush=True)
mb = float(D.b.mean())
def feats(d, kind):
    X = [lg(d.a.to_numpy())]
    if kind == "b":
        X += [d.b.fillna(mb).to_numpy(), d.b.isna().to_numpy().astype(float)]
    if kind == "c":
        X += [lg(d.c.to_numpy())]
    return np.column_stack(X)
for fit_, read_ in (("JF", "M3"), ("M3", "JF")):
    df, dr = D[D.src == fit_], D[D.src == read_]; out = [f"fit {fit_} -> read {read_}: (a) {roc_auc_score(dr.vanish, dr.a):.3f}"]
    for kind in ("b", "c"):
        lr = LogisticRegression(C=1e6, max_iter=2000).fit(feats(df, kind), df.vanish); out.append(f"(a)+({kind}) {roc_auc_score(dr.vanish, lr.predict_proba(feats(dr, kind))[:, 1]):.3f}")
    print(" | ".join(out), flush=True)
