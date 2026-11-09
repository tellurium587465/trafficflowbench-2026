"""G109 (10/29, strategy lane 11, ledger #50): a hazard for the end of a visible jam from (site, age, time of day, start
time), learned on the training months only, mixed into the final A ongoing probabilities on M3 v3.
The generator's jams are per-site rectangles that end all at once at a random time (audit/b13*): the end does not follow the
upstream flow (AUC 0.46) but depends on the site and the time of day (AUC 0.62 age -> 0.87 + site, tod, start tod). The
final A models see 1 h of history, no time of day, no site identity.
Model: training-month events at the known sites (b13_sites, B13's fill and tracking, all events >= 2 rows), rows (event, age
a, h = 1..6) -> y = alive at a + h; LightGBM (site categorical, age, tod at T, start tod, h).
M3: per ongoing window and k, each run of last_queue whose head is a known site head +-1: the start row is estimated by
walking back from T - 1 over the day's rows at the head link (rows hidden by the simulated blanks of (day, k) and masked
cells are skipped; the walk stops at the first observed free row), S_h = model(site, T - start, T % 288, start % 288, h);
p[h - 1, run] <- (1 - beta) p + beta S_h after the cone, before the evidence weight / best_set / far gate.
H1 beta = 0.5 (primary), H2 beta = 0.3; A (beta = 0) must reproduce G93's lgb_cone.
Diagnostics (dev): AUC for 'the run is all gone at T+30' of the A probability (run mean at T+30) and of S_6.
Writes work/t2/g109_end_hazard.csv for judge.py --cond queue_ongoing.
usage: TFB_M3VER=v3 py g109_end_hazard.py"""
import os, sys
os.environ.setdefault("TFB_M3VER", "v3")
sys.path.insert(0, r"C:/Users/user/tfb/audit")
src = open(r"C:/Users/user/tfb/review/g108_rectangles.py", encoding="utf-8").read().split("\ncols = {c: {} for c")[0]
exec(compile(src, "g108", "exec"))                              # G93's final A ongoing flow, runs(), og_pred(), SITE
from cachelib import load
from common import PANELS
from b13_jam_shapes import fill
from sklearn.metrics import roc_auc_score

SID = {k: i for i, k in enumerate(sorted(SITE))}
FEAT = ["site", "age", "tod", "tod0", "h"]


def training_rows():
    rows = []
    for p in PANELS:
        heads = {h for (pp, h) in SITE if pp == p}
        if not heads:
            continue
        P = load(p, "train"); C = np.where(np.isnan(P.V), np.nan, (P.V <= P.vcut[None, :]).astype(float)); B = fill(C)
        active, done = {}, []
        for t in range(P.T):                                    # B13's tracking
            new = {}
            for a, b in runs(B[t]):
                prev = next((h for h in (b, b - 1, b + 1) if h in active and h not in [e["hid"] for e in new.values()]), None)
                ev = active.pop(prev) if prev is not None else {"t0": t, "heads": [], "hid": None}
                ev["heads"].append(b); ev["hid"] = b; new[b] = ev
            done += list(active.values()); active = new
        for ev in done:
            n = len(ev["heads"]); hm = int(np.bincount(ev["heads"]).argmax())
            if n < 2 or hm not in heads or (ev["t0"] % 288) + n >= 288:
                continue
            sid, t0 = SID[(p, hm)], ev["t0"]
            for a in range(1, n):
                for h in range(1, 7):
                    rows.append((sid, a, (t0 + a) % 288, t0 % 288, h, a + h <= n - 1))
        load.cache_clear()
    return pd.DataFrame(rows, columns=FEAT + ["y"])


R = training_rows(); print("training rows", len(R), "alive share", round(R.y.mean(), 3), flush=True)
HZ = lgb.train(dict(objective="binary", learning_rate=0.05, num_leaves=63, min_data_in_leaf=200, verbose=-1, num_threads=8),
               lgb.Dataset(R[FEAT], R.y, categorical_feature=["site"]), 400)
HZ.save_model(W + "g109_end_hazard.txt")


def start_row(m, head, k):
    P, _, q = m3v2._panel(m["panel"]); t = m["t"]; d0 = t - (t % 288); hid = V._hidden_today(m, k); st = None
    for r in range(t - 1, d0 - 1, -1):
        if r in hid or not np.isfinite(P.V[r, head]):
            continue
        if q[r, head]:
            st = r
        else:
            break
    return st if st is not None else d0


BETA = {"A": 0.0, "H1": 0.5, "H2": 0.3}
cols = {c: {} for c in BETA}; dmax = 0.0; dg = []
for i, w in enumerate(G.WID):
    m = G.META[w]
    if m["cond"] != "queue_ongoing":
        continue
    Xw = np.asarray(G.X[G.OFF[i]:G.OFF[i + 1]]); pl = FS.lgb_ongoing(Xw); R_ = runs(np.asarray(Xw[:, G.CI], float) == 1)
    dev = not m3v2.is_holdout(np.array([m["day"]]))[0]; sc = {c: 0.0 for c in BETA}
    for k, pr in m3v2.kdist(m["day"]).items():
        p0 = (0.8 * X1[(w, k)] + 0.2 * pl) * cone_f(m, k); SH = []
        for a, b in R_:
            h = next((hh for hh in (b, b - 1, b + 1) if (m["panel"], hh) in SITE), None)
            if h is None:
                continue
            st = start_row(m, h, k); t = m["t"]
            F = pd.DataFrame({"site": SID[(m["panel"], h)], "age": t - st, "tod": t % 288, "tod0": st % 288, "h": np.arange(1, 7)})
            SH.append((a, b, HZ.predict(F[FEAT])))
        for c, beta in BETA.items():
            p = p0.copy()
            for a, b, s_ in SH:
                p[:, a:b + 1] = (1 - beta) * p[:, a:b + 1] + beta * s_[:, None]
            sc[c] += pr * G.iou(og_pred(p, m, k, None, R_), m)
        if dev:
            y, mk = m["y"].astype(bool), m["m"].astype(bool)
            for a, b, s_ in SH:
                o = mk[a:b + 1, 5]
                if o.sum() >= 1:
                    dg.append((pr, y[a:b + 1, 5][o].mean() == 0, p0[5, a:b + 1].mean(), s_[5]))
    for c in sc:
        cols[c][w] = sc[c]
    dmax = max(dmax, abs(cols["A"][w] - REF.loc[w, "lgb_cone"]))
print(f"identity: A vs G93 lgb_cone: max |d| {dmax:.2e}")
g = np.array(dg, float)
if len(g):
    wt, y, pa, s6 = g.T
    print(f"dev visible runs at known sites (k-weighted rows {len(g)}, all gone at T+30 {np.average(y, weights=wt):.3f}): "
          f"AUC A {roc_auc_score(y, -pa, sample_weight=wt):.3f} | S_6 {roc_auc_score(y, -s6, sample_weight=wt):.3f} | "
          f"mix 0.5 {roc_auc_score(y, -(0.5 * pa + 0.5 * s6), sample_weight=wt):.3f}")
D = V.frame(cols); D = D[D.cond == "queue_ongoing"]; D.to_csv(W + "g109_end_hazard.csv", index=False)
dv = ~m3v2.is_holdout(D.day)
print("dev ongoing", int(dv.sum()), " | ".join(f"{c} {D[dv][c].mean():.4f}" for c in BETA))
