"""G119 (10/30, ledger #72, strategy J8): do the T2 post-processing constants chosen on March hold in other months?
Train months 2031-01 / 2031-02, the selector-like windows (featcache2_noT_af is_sel), month-CV OOF of the no-memory NN
(q25_nomem seeds 3, 4). The March post-processing as is, re-implemented month-generic (the code path of m3v2 /
t2_m3v2_gate / g55 / g108; identity-checked on March first):
  onset  : raw (best_set) | E evidence x0.25 | P physics onset (adaptive a) | PE
  ongoing: raw (best_set) | S seen x0.85 | SG + far gate D=3 + contiguous | SCG + cone (g55, released by today's evidence)
  B = E / SG (final B's April set), A = PE / SCG (final A's, without X1 / E4 / tfm / LightGBM 0.2)
k = the expectation over the official distribution truncated at the day of the month (cap 3), selector-simulated blanks.
Train months: physics model, rarity table and adaptive a leave the month out. Visibility: "mask" (primary) = the Task 1
blank hidden on every row except the window's T-12..T-1 (as the NN inputs and the official windows), "full" = nothing
hidden. Estimator: per condition the paired mean, T2 = mean of the two, SE = cluster (panel x day) bootstrap, unweighted;
March dev (hold-out days not read) under the same functions (NOMEM q24f seeds 0-3, physon_base) for the comparison.
usage: TFB_M3VER=v3 py g119_month_postproc.py [time | identity | run]"""
import os, sys, time, types, pickle, functools
os.environ["TFB_T2CACHE"] = "2_noT_af"; os.environ.setdefault("TFB_M3VER", "v3")
sys.path.insert(0, r"C:/Users/user/tfb/src"); MODE = (sys.argv[1:] or ["run"])[0]; sys.argv = [sys.argv[0], "0,1,2,3"]
import numpy as np, pandas as pd, lightgbm as lgb
import m3v2, t2_phys_onset as PO, t2_m3_gate as G
from t2_eval_oof import T
from t2_model import best_set
from cachelib import load
from common import NN

W = G.W; TM = time.perf_counter; CUR = {}
XN = np.load(f"{NN}/nn_data_afdem/X.npy", mmap_mode="r"); OFFN = np.load(f"{NN}/nn_data_afdem/off.npy")
assert len(OFFN) - 1 == T.NWIN, "nn_data windows differ from the evaluator windows"
MON = np.array([str(d)[:7] for d in T.META["date"]]); MONTHS = ("2031-01", "2031-02")
CONDS = ("queue_onset", "queue_ongoing")


def tick(name, t0):
    CUR[name] = CUR.get(name, 0.0) + TM() - t0


class Ctx:
    """a panel as the post-processing sees it: vis = the visible speeds (NaN = hidden); full (optional) = the layer the
    window's own T-12..T-1 rows are read from"""
    def __init__(self, key, P, vis, full=None):
        self.key, self.n = key, P.n
        self.ns = types.SimpleNamespace(V=vis, E=P.E, T=P.T, times=P.times, n=P.n)
        self.Qm = (np.nan_to_num(vis, nan=1e9) <= P.vcut[None, :]) & P.E           # t2_windows.queue_matrix
        self.q = (vis <= P.vcut[None, :]) & np.isfinite(vis); self.obs = np.isfinite(vis)  # m3v2._panel
        self.qf = None if full is None else (full <= P.vcut[None, :]) & np.isfinite(full)
        self.of = None if full is None else np.isfinite(full)

    def rows(self, rr, t, kind="q"):
        a = (self.q if kind == "q" else self.obs)[rr]
        if self.qf is not None:
            rr = np.asarray(rr); h = (rr >= t - 12) & (rr < t)
            if h.any():
                a = a.copy(); a[h] = (self.qf if kind == "q" else self.of)[rr[h]]
        return a


CTX = {}


@functools.lru_cache(maxsize=None)
def sim_blank(ckey, day, k, t):                                  # m3v2.sim_blank_rows
    c = CTX[ckey]; t0 = TM(); start = (day - max(k, 0)) * 288
    on, og = m3v2.scan_until(c.ns, c.Qm, max(start, 12), t); rows = set()
    for s in on + og:
        if s < t:
            rows.update(range(s + 1, min(s + 19, t)))
    tick("scan", t0); return frozenset(rows)


def seen(c, t, k, lo, hi, cap=3):                               # m3v2.seen (earlier days only: never the window's T-12..T-1)
    s = np.zeros(c.n, bool); nd = 0; hid = sim_blank(c.key, t // 288, min(cap, k), t); t0 = TM()
    for back in range(1, min(cap, k) + 1):
        dd = t - back * 288
        if dd + lo < 0:
            break
        rr = [r for r in range(dd + lo, dd + hi) if r not in hid]
        if rr:
            s |= c.q[rr].any(axis=0)
        nd += 1
    tick("seen", t0); return s, nd


def decode(pb):
    t0 = TM(); s = best_set(pb.ravel(), np.ones(pb.size, bool)).reshape(pb.shape); tick("best_set", t0); return s


def onset_pred(c, p_, t, k, rule):                              # t2_m3v2_gate.onset_pred
    pb = p_.copy()
    if rule != "none":
        s, nd = seen(c, t, k, -6, 12)
        if nd:
            pb[-1] = pb[-1] * np.where(G.dil(s, 2), 1.0, 0.25)
    return decode(pb)


def today_ev(c, t, k):
    d0 = t - (t % 288) + 60; hid = sim_blank(c.key, t // 288, min(3, k), t)
    rr = [r for r in range(d0, t) if r not in hid]
    return c.rows(rr, t).any(axis=0) if rr else np.zeros(c.n, bool)


def far_gate(c, sel, t, k, lq):                                 # t2_m3v2_gate.far_gate
    ev = today_ev(c, t, k); s2, _ = seen(c, t, k, -18, 12); t0 = TM(); ev = G.dil(ev | s2, 2)
    near = np.zeros(c.n, bool)
    for j in np.flatnonzero(lq):
        near[max(0, j - 3):j + 4] = True
    C = np.zeros_like(sel)
    for r_ in range(sel.shape[0]):
        j = 0
        while j < sel.shape[1]:
            if sel[r_, j]:
                e = j
                while e + 1 < sel.shape[1] and sel[r_, e + 1]:
                    e += 1
                if near[j:e + 1].any():
                    C[r_, j:e + 1] = True
                j = e + 1
            else:
                j += 1
    out = sel & ~(sel & ~near[None, :] & ~ev[None, :] & ~C); tick("far_gate", t0); return out


def ongoing_pred(c, p_, t, k, gate, lq):                        # t2_m3v2_gate.ongoing_pred
    s, nd = seen(c, t, k, -18, 12)
    pb = p_ * np.where(G.dil(s, 1), 1.0, 0.85)[None, :] if nd else p_.copy()
    sel = decode(pb)
    return far_gate(c, sel, t, k, lq) if gate else sel


WSTEP = 25.0 * 5 / 60


def runs(v):
    out, j = [], 0
    while j < len(v):
        if v[j]:
            e = j
            while e + 1 < len(v) and v[e + 1]:
                e += 1
            out.append((j, e)); j = e + 1
        else:
            j += 1
    return out


def cone_parts(c, t, lq, length):                               # g55.parts (cone only), rows T-6..T-1
    t0 = TM(); n = c.n; up = -1; xc = np.cumsum(length) - length / 2
    rr = np.arange(t - 6, t); O = c.rows(rr, t, "obs"); Q = c.rows(rr, t, "q"); cone = np.ones((6, n))
    for a, b in runs(lq):
        head = b if up == -1 else a; pts = []
        for i_, r in enumerate(rr):
            j, last = head, None
            while 0 <= j < n and not (O[i_, j] and not Q[i_, j]):
                if O[i_, j] and Q[i_, j]:
                    last = j
                j += up
            if last is not None:
                pts.append((r - t, abs(xc[last] - xc[head])))
        if len(pts) >= 3:
            xs, ls = np.array(pts, float).T; s, L0 = np.polyfit(xs, ls, 1); s = float(np.clip(s, -WSTEP, WSTEP))
            L0 = float(np.mean(ls - s * xs))
        else:
            s = 0.0; tl = a if up == -1 else b; L0 = abs(xc[tl] - xc[head])
        j_end = head
        while 0 <= j_end + up < n and not lq[j_end + up] or (0 <= j_end + up < n and a <= j_end + up <= b):
            j_end += up
        js = np.arange(head, j_end + up, up) if j_end != head else np.array([head])
        d = np.abs(xc[js] - xc[head])
        for h in range(1, 7):
            cone[h - 1, js[d > L0 + WSTEP * h]] = 0.5
    tick("cone", t0); return cone


def cone_f(c, cone, t, k):                                      # g108.cone_f
    ev_ = today_ev(c, t, k); s2, _ = seen(c, t, k, -18, 12)
    return np.where(G.dil(ev_ | s2, 2)[None, :], 1.0, cone)


def iou(sel, y, e, cond):                                       # t2_m3_gate.iou
    if cond == "queue_onset":
        sel, y, e = sel[-1:], y[-1:], e[-1:]
    u = ((sel | y) & e).sum(); return 1.0 if u == 0 else float((sel & y & e).sum() / u)


def score_window(c, p, t, cond, y, e, lq, dom, length, ph=None, a=0.0):
    out = {}
    cone = cone_parts(c, t, lq, length) if cond == "queue_ongoing" else None
    for k, pr in m3v2.kdist(dom, cap=3).items():
        if cond == "queue_onset":
            pp = p.copy(); pp[5] = (1 - a) * p[5] + a * ph
            cols = {"raw": onset_pred(c, p, t, k, "none"), "E": onset_pred(c, p, t, k, "cur"),
                    "P": onset_pred(c, pp, t, k, "none"), "PE": onset_pred(c, pp, t, k, "cur")}
        else:
            cols = {"raw": decode(p), "S": ongoing_pred(c, p, t, k, False, lq), "SG": ongoing_pred(c, p, t, k, True, lq),
                    "SCG": ongoing_pred(c, p * cone_f(c, cone, t, k), t, k, True, lq)}
        t0 = TM()
        for nm, sel in cols.items():
            out[nm] = out.get(nm, 0.0) + pr * iou(sel, y, e, cond)
        tick("iou", t0)
    out["B"] = out["E"] if cond == "queue_onset" else out["SG"]; out["A"] = out["PE"] if cond == "queue_onset" else out["SCG"]
    return out


def boot(D, a, b, n_boot=2000, seed=0):
    """paired b - a: per-condition means, their SEs, T2 (mean of the conditions present) and its SE; cluster = panel x day"""
    rng = np.random.default_rng(seed); D = D.dropna(subset=[a, b]); d = (D[b] - D[a]).to_numpy()
    cs = [c_ for c_ in CONDS if (D.cond == c_).any()]
    cl = pd.factorize(D.panel.astype(str) + "_" + D.day.astype(str))[0]; ncl = cl.max() + 1
    ci = np.array([cs.index(c_) for c_ in D.cond]); M = np.zeros((ncl, len(cs))); N = np.zeros((ncl, len(cs)))
    np.add.at(M, (cl, ci), d); np.add.at(N, (cl, ci), 1)
    R = rng.multinomial(ncl, np.full(ncl, 1.0 / ncl), size=n_boot); bs = (R @ M) / np.maximum(R @ N, 1e-12)
    per = M.sum(0) / N.sum(0)
    return {c_: (per[i], bs[:, i].std()) for i, c_ in enumerate(cs)}, (per.mean(), bs.mean(axis=1).std())


# ---------------------------------------------------------------- March (identity + comparison)
def march(limit=None):
    phys = lgb.Booster(model_file=W + "physon_base.txt"); ev = PO.month_events("validation", PO.rarity_table()); rows = []
    for w in G.WID:
        m = G.META[w]
        if m3v2.is_holdout(np.array([m["day"]]))[0]:
            continue
        key = ("val", m["panel"])
        if key not in CTX:
            t0 = TM(); P = load(m["panel"], "validation"); CTX[key] = Ctx(key, P, P.V); tick("ctx", t0)
        c = CTX[key]; i = G.POS[w]; Xw = np.asarray(G.X[G.OFF[i]:G.OFF[i + 1]]); lq = np.asarray(Xw[:, G.CI], float) == 1
        ph = a = None
        if m["cond"] == "queue_onset":
            t0 = TM(); ph = phys.predict(PO.feats(Xw)); a = PO.adaptive_a(ev, m["day"]); tick("phys", t0)
        r = score_window(c, G.NOMEM[w], m["t"], m["cond"], m["y"].T.astype(bool), m["m"].T.astype(bool), lq, m["day"],
                         load(m["panel"], "validation").length, ph, a or 0.0)
        rows.append(dict(w=w, month="2031-03", vis="released", cond=m["cond"], panel=m["panel"], day=m["day"], **r))
        if limit and len(rows) >= limit:
            break
    return pd.DataFrame(rows)


def identity(Dm):
    ref = pd.read_csv(W + "m3v2_gatev3.csv").set_index("w"); d = 0.0; n = 0
    for r in Dm.itertuples():
        pairs = (("E", "nomem"), ("raw", "ev_none")) if r.cond == "queue_onset" else (("SG", "nomem"), ("S", "gate_off"))
        for mine, theirs in pairs:
            d = max(d, abs(getattr(r, mine) - ref.loc[r.w, theirs])); n += 1
    return d, n


# ---------------------------------------------------------------- train months (leave the month out)
F_CACHE = W + "g119_physfeats.pkl"


def phys_feats():
    """PO.feats of every onset window with a T+30 queue (the physics model's training rows, G70 / physon_base set)"""
    if os.path.exists(F_CACHE):
        return pickle.load(open(F_CACHE, "rb"))
    t0 = TM(); F = {}
    ON = [k for k in range(T.NWIN) if str(T.META["cond"][k]) == "queue_onset" and (T.LAB[k][0][5] & T.LAB[k][1][5]).any()]
    for k in ON:
        F[k] = PO.feats(np.asarray(XN[OFFN[k]:OFFN[k + 1]])).astype(np.float32)
    tick("phys_feats", t0); pickle.dump(F, open(F_CACHE, "wb")); return F


def phys_lomo(F, mo):
    t0 = TM(); tr = [k for k in F if MON[k] != mo]
    X = np.vstack([F[k][T.LAB[k][1][5].astype(bool)] for k in tr])
    y = np.concatenate([T.LAB[k][0][5][T.LAB[k][1][5].astype(bool)] for k in tr]).astype(int)
    mdl = lgb.train(PO.PRM, lgb.Dataset(X, y), 400); tick("phys_train", t0); return mdl


def rarity_lomo(mo):                                            # PO.rarity_table without the month
    cnt, den = {}, {}
    for k in range(T.NWIN):
        if T.META["is_sel"][k] and str(T.META["cond"][k]) == "queue_onset" and MON[k] != mo:
            lab, el = T.LAB[k]; key = (str(T.META["panel"][k]), pd.Timestamp(str(T.META["date"][k])[:10]).dayofweek >= 5, (int(T.META["t"][k]) % 288) // 24)
            cnt[key] = cnt.get(key, 0) + (lab[5] & el[5]).astype(float); den[key] = den.get(key, 0) + 1
    return {k: cnt[k] / den[k] for k in cnt}


def month_events(c, p, P, mo, F_):                              # PO.month_events on one train month, day0 = day of the month
    r0 = int(np.flatnonzero(np.char.startswith(P.times.astype(str), mo))[0]); r1 = r0 + int(np.char.startswith(P.times.astype(str), mo).sum())
    anyq = c.q.any(axis=1); last = -10 ** 9; ev = []
    for t in range(r0 + 17, r1):
        if anyq[t] and not anyq[t - 17:t].any() and t - last >= 24:
            last = t; o = t - 6; key = (p, pd.Timestamp(P.times[o][:10]).dayofweek >= 5, (o % 288) // 24)
            f = F_.get(key, np.zeros(P.n)); ev.append(((t - r0) // 288, bool((f[np.flatnonzero(c.q[t])] <= 0.1).all())))
    return ev


def train_months(limit=None):
    OOF = [pickle.load(open(W + f"oof_q25_nomem_s{s}.pkl", "rb")) for s in (3, 4)]
    F = phys_feats() if limit is None else None
    rows = []; CI = G.CI
    for mo in MONTHS:
        mdl = phys_lomo(F, mo) if F is not None else lgb.Booster(model_file=W + "physon_base.txt")   # time mode: the base model
        F_ = rarity_lomo(mo); ks = np.flatnonzero(T.META["is_sel"] & (MON == mo))
        for p in sorted({str(T.META["panel"][k]) for k in ks}):
            t0 = TM(); P = load(p, "train"); tick("load", t0)
            for vis in ("mask", "full"):
                t0 = TM(); key = (mo, p, vis); CTX.clear(); sim_blank.cache_clear()
                CTX[key] = Ctx(key, P, np.where(P.blank, np.nan, P.V), P.V) if vis == "mask" else Ctx(key, P, P.V); c = CTX[key]; tick("ctx", t0)
                t0 = TM(); ev = month_events(c, p, P, mo, F_); tick("events", t0)
                for k in ks[np.array([str(T.META["panel"][k]) == p for k in ks])]:
                    t = int(T.META["t"][k]); cond = str(T.META["cond"][k]); assert P.times[t][:10] == str(T.META["date"][k])[:10]
                    dom = int(P.times[t][8:10]) - 1; lab, el = T.LAB[k]; Xw = np.asarray(XN[OFFN[k]:OFFN[k + 1]])
                    lq = np.asarray(Xw[:, CI], float) == 1; p_ = (OOF[0][k] + OOF[1][k]) / 2; ph = None; a = 0.0
                    if cond == "queue_onset":
                        t0 = TM(); ph = mdl.predict(PO.feats(Xw)); a = PO.adaptive_a(ev, dom); tick("phys", t0)
                    r = score_window(c, p_, t, cond, lab.astype(bool), el.astype(bool), lq, dom, P.length, ph, a)
                    rows.append(dict(w=int(k), month=mo, vis=vis, cond=cond, panel=p, day=dom, a=a, **r))
                    if limit and len(rows) >= limit:
                        return pd.DataFrame(rows)
            load.cache_clear()
        print(mo, "done", len(rows), "rows", {k: round(v, 1) for k, v in CUR.items()}, flush=True)
    return pd.DataFrame(rows)


def report(D, name):
    out = [f"{name}: windows onset {int((D.cond == 'queue_onset').sum())} / ongoing {int((D.cond == 'queue_ongoing').sum())} | "
           f"IoU raw onset {D[D.cond == 'queue_onset'].raw.mean():.4f} ongoing {D[D.cond == 'queue_ongoing'].raw.mean():.4f}"]
    for lab_, a, b, conds in (("B set (E / SG) - raw", "raw", "B", CONDS), ("A set (PE / SCG) - raw", "raw", "A", CONDS),
                              ("onset E - raw", "raw", "E", CONDS[:1]), ("onset P - raw", "raw", "P", CONDS[:1]), ("onset PE - E (A's extra)", "E", "PE", CONDS[:1]),
                              ("ongoing S - raw", "raw", "S", CONDS[1:]), ("ongoing SG - S (gate)", "S", "SG", CONDS[1:]),
                              ("ongoing SCG - SG (A's extra: cone)", "SG", "SCG", CONDS[1:])):
        per, (t2, se) = boot(D[D.cond.isin(conds)], a, b)
        pc = " | ".join(f"{c_.split('_')[1]} {m_:+.4f} (SE {s_:.4f}, z {m_ / s_ if s_ > 0 else 0:+.1f})" for c_, (m_, s_) in per.items())
        out.append(f"  {lab_:36s} " + (f"T2 {t2:+.4f} (SE {se:.4f}, z {t2 / se if se > 0 else 0:+.1f}) | " if len(conds) == 2 else "") + pc)
    return "\n".join(out)


if __name__ == "__main__":
    if MODE == "time":                                          # per-stage time on a few windows, then the projection
        CUR.clear(); t0 = TM(); Dm = march(limit=6); tm = TM() - t0
        print(f"March: {len(Dm)} windows {tm:.1f}s", {k: round(v, 2) for k, v in CUR.items()}); d, n = identity(Dm); print(f"identity on those: max |d| {d:.2e} over {n}")
        CUR.clear(); t0 = TM(); Dt = train_months(limit=6); tt = TM() - t0
        print(f"train: {len(Dt)} windows {tt:.1f}s", {k: round(v, 2) for k, v in CUR.items()})
        t0 = TM(); k0 = [k for k in range(T.NWIN) if str(T.META["cond"][k]) == "queue_onset"][:200]
        for k in k0:
            PO.feats(np.asarray(XN[OFFN[k]:OFFN[k + 1]]))
        non = sum(1 for k in range(T.NWIN) if str(T.META["cond"][k]) == "queue_onset" and (T.LAB[k][0][5] & T.LAB[k][1][5]).any())
        print(f"phys feats: {(TM() - t0) / 200 * 1e3:.1f} ms per window x {non} training onset windows = {(TM() - t0) / 200 * non:.0f}s (cached once)")
        sys.exit()
    t0 = TM(); Dm = march(); d, n = identity(Dm)
    print(f"March dev: {len(Dm)} windows in {TM() - t0:.0f}s | identity vs m3v2_gatev3.csv (E / raw onset, SG / S ongoing): max |d| {d:.2e} over {n}", flush=True)
    if d > 1e-9:
        print("IDENTITY FAILED: stop (the month-generic path is not the instrument)"); sys.exit(1)
    if MODE == "identity":
        sys.exit()
    print(report(Dm, "2031-03 dev (released)"), flush=True)
    CUR.clear(); t0 = TM(); Dt = train_months(); print(f"train months in {TM() - t0:.0f}s", {k: round(v, 1) for k, v in CUR.items()})
    D = pd.concat([Dm, Dt], ignore_index=True); D.to_csv(W + "g119_month_postproc.csv", index=False)
    for mo in MONTHS:
        for vis in ("mask", "full"):
            print(report(Dt[(Dt.month == mo) & (Dt.vis == vis)], f"{mo} ({vis})"), flush=True)
