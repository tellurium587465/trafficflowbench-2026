"""G332 (11/05, ledger #263c, measurement only): G331 with the oracle count k_span = per true run (the true scored cells grouped while the
cells between them are all unscored), the links from its first to its last true cell, unscored ones included, summed; the top k_span of C.
From G331: Candidate cells C = the T+30 cells of the links observed at
least once in the window's history rows T-12 .. T-1 or on the previous day within +-30 min of the window's T+30 time (the horizon's score
eligibility is data stamped after T and is not used). Oracle bound: the top k of C (k = the true scored count). Arms: k_hat = max(1,
round(sum of p over C)) (i), max(1, round(c x sum)) (ii) with c = sum of the true count / sum of sum p on January's onset windows (a training
month, the selection side); the top k_hat of C. Instruments as G328 (M3 dev = A17 onset rows of G298; January / February = capB0 rows of
G298; June = cvf rows of G329). Selection on M3 odd days + January, the read on M3 even days + February, June recorded.
Line: both read halves > 0, merged z >= +1. usage: TFB_M3VER=v3 py -B -X utf8 g331_onset_khat.py"""
import sys
src = open(r"C:/Users/user/tfb/review/g328_oracle_map.py", encoding="utf-8").read()
exec(compile(src.split("# --- M3 dev (A17)")[0], "g328h", "exec"))
from cachelib import load as cload, T2_PANELS


def cand(Vwin, Vprev):
    """links observed in the history rows or the previous day's slot"""
    o = np.isfinite(Vwin).any(axis=0)
    if Vprev is not None and len(Vprev):
        o |= np.isfinite(Vprev).any(axis=0)
    return o


def arms(row, y, e, C, c_):
    yt = y & e; k = int(yt.sum()); r = np.where(C, row, -1.0); dec = best_set(row, np.ones(row.size, bool)); out = dict(base=iou1(dec, y, e))
    def top(kk):
        s_ = np.zeros(row.size, bool); s_[np.argsort(-r)[:max(int(min(kk, C.sum())), 0)]] = True; return s_ & C
    pos = np.flatnonzero(yt); kspan = 0
    if len(pos):
        g0 = pos[0]
        for i_ in range(1, len(pos) + 1):
            if i_ == len(pos) or e[pos[i_ - 1] + 1:pos[i_]].any():
                kspan += pos[i_ - 1] - g0 + 1
                if i_ < len(pos):
                    g0 = pos[i_]
    sp = float(row[C].sum()); out["oracleC"] = iou1(top(kspan), y, e) if k else out["base"]; out["armi"] = iou1(top(max(1, round(sp))), y, e); out["armii"] = iou1(top(max(1, round(c_ * sp))), y, e)
    return out, k, sp


VM = {}
for p in T2_PANELS:
    P = cload(p, "validation"); VM[p] = P.V.copy(); cload.cache_clear()
D8 = pickle.load(open(W + "g298_on_pretemp.pkl", "rb")); D9 = pickle.load(open(W + "g329_jun_onset_rows.pkl", "rb"))


def ctx_train(k):
    p = str(T_.META["panel"][k]); t = int(T_.META["t"][k]); P = h.load(p, "train"); Vw = P.V[t - 12:t]; tp = t - 288 + 6; Vp = np.where(P.blank, np.nan, P.V)[max(tp - 6, 0):tp + 7] if t - 288 >= 0 else None
    return cand(Vw, Vp)


ks, sps = [], []
for k, v in D8["J"].items():
    if MON[k] != "2031-01":
        continue
    C = ctx_train(k); yt = v["y"] & v["e"]
    for pr, row in v["C"]:
        ks.append(pr * yt.sum()); sps.append(pr * row[C].sum())
C_ = float(np.sum(ks) / np.sum(sps)); print(f"c from January's onset windows: sum true count / sum of sum p over C = {C_:.3f}", flush=True)
rows = []
for w, v in D8["M"].items():
    m = G.META[w]; t = m["t"]; Vw = VM[m["panel"]][t - 12:t]; tp = t - 288 + 6; C = cand(Vw, VM[m["panel"]][max(tp - 6, 0):tp + 7] if t - 288 >= 0 else None)
    r = dict(src="M3", w=w, day=m["day"], panel=m["panel"], cond="queue_onset", hour=(t % 288) // 12, base=0.0, oracleC=0.0, armi=0.0, armii=0.0, ncand=int(C.sum()))
    for pr, row in v["A"]:
        o, _, _ = arms(row, v["y"], v["e"], C, C_)
        for k_, v_ in o.items():
            r[k_] += pr * v_
    rows.append(r)
for lab, srcd in (("JF", D8["J"]), ("JUN", D9)):
    for k, v in srcd.items():
        C = ctx_train(k); r = dict(src=lab, w=int(k), month=MON[k], panel=str(T_.META["panel"][k]), day=int(T_.META["t"][k]) // 288, base=0.0, oracleC=0.0, armi=0.0, armii=0.0, ncand=int(C.sum()))
        for pr, row in v["C"]:
            o, _, _ = arms(row, v["y"], v["e"], C, C_)
            for k_, v_ in o.items():
                r[k_] += pr * v_
        rows.append(r)
    h.load.cache_clear()
D = pd.DataFrame(rows); D.to_csv(W + "g332_onset_kspan.csv", index=False)
P8 = pd.read_csv(W + "g280_m3.csv").set_index("w").pc4; dm = D[D.src == "M3"]; print("identity M3 base against g280 pc4:", float(np.abs(dm.set_index("w").base - P8.reindex(dm.w).to_numpy()).max()), flush=True)
for s_ in ("M3", "JF", "JUN"):
    d = D[D.src == s_]; print(f"{s_}: {len(d)} windows, base {d.base.mean():.4f} | oracle top-k_span in C {d.oracleC.mean() - d.base.mean():+.4f} (G328 O_size over the scored cells for comparison) | arm i {d.armi.mean() - d.base.mean():+.4f} | arm ii {d.armii.mean() - d.base.mean():+.4f} | candidate links median {d.ncand.median():.0f}", flush=True)
bound = max(D[D.src == s_].oracleC.mean() - D[D.src == s_].base.mean() for s_ in ("M3", "JF", "JUN"))
print('arms are read only if the bound >= +0.005; (i)(ii) printed above are the k_hat arms of G331', flush=True)
if bound < 0.005:
    print(f"oracle bound in C below +0.005 on every instrument ({bound:+.4f}): the arms are not read (#263 closes)", flush=True); sys.exit()
rg = np.random.default_rng(0)
def m3s(d, a):
    r_ = m3v2.summary(d, "base", a, "private"); return r_["t2_weighted"], r_["se"]
def jfs(d, a):
    cl = pd.factorize(d.panel + "_" + d.day.astype(str))[0]; Rm = rg.multinomial(cl.max() + 1, np.full(cl.max() + 1, 1 / (cl.max() + 1)), size=2000); x = (d[a] - d.base).to_numpy(); return x.mean(), float(((Rm @ np.bincount(cl, x)) / (Rm @ np.bincount(cl))).std())
def mg(x, y):
    w1, w2 = 1 / max(x[1], 1e-9) ** 2, 1 / max(y[1], 1e-9) ** 2; m_ = (w1 * x[0] + w2 * y[0]) / (w1 + w2); return m_, (w1 + w2) ** -0.5
M3d = D[D.src == "M3"].reset_index(drop=True); odd = M3d[(M3d.day + 1) % 2 == 1].reset_index(drop=True); evn = M3d[(M3d.day + 1) % 2 == 0].reset_index(drop=True)
jan = D[(D.src == "JF") & (D.month == "2031-01")].reset_index(drop=True); feb = D[(D.src == "JF") & (D.month == "2031-02")].reset_index(drop=True); sel = {}
for a in ("armi", "armii"):
    x, y = m3s(odd, a), jfs(jan, a); sel[a] = mg(x, y); print(f"select {a}: M3 odd {x[0]:+.5f} ({x[1]:.5f}) | Jan {y[0]:+.5f} ({y[1]:.5f}) | merged {sel[a][0]:+.5f} (z {sel[a][0] / sel[a][1]:+.2f})", flush=True)
best = max(sel, key=lambda a: sel[a][0])
if sel[best][0] <= 0:
    print("selection: no arm above best_set; #263 closes", flush=True); sys.exit()
x, y = m3s(evn, best), jfs(feb, best); m_, s_ = mg(x, y); j = D[D.src == "JUN"]
print(f"read {best}: M3 even {x[0]:+.5f} (SE {x[1]:.5f}) | Feb {y[0]:+.5f} (SE {y[1]:.5f}) | merged {m_:+.5f} (z {m_ / s_:+.2f}) | June (record) {(j[best] - j.base).mean():+.5f} | line: {x[0] > 0 and y[0] > 0 and m_ / s_ >= 1}", flush=True)
