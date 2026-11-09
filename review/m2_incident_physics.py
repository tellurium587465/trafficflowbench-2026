"""M2: incident-type (non-recurrent) queues in train vs recurrent ones -- location-independent physics.
Per T2 panel (train only, unmasked observation layer, no labels / no test data):
 episodes = space-time connected queued segments (v <= vcut), >= 3 steps, >= 2 links at some point.
 head site frequency f = share of days with an episode headed within +-2 links -> rare (f<0.05) vs recurrent (f>=0.2).
 discharge ratio r = flow on the first free-flow links downstream of the head / min fd capacity there.
 tail kinematics over the first 30 min: Rankine-Hugoniot w_s = (q_up - q_out)/(k_up - k_q) vs actual.
 dissolution: head speed over the last 30 min (recovery wave, expected ~ w).
Usage: python m2_incident_physics.py [PANEL ...]  (CPU, one panel in memory at a time, < 1 GB)"""
import sys, numpy as np, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, r"C:/Users/user/tfb/src")
from cachelib import load, T2_PANELS


def runs(row):
    d = np.diff(np.concatenate([[0], row.astype(np.int8), [0]]))
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1) - 1))


def episodes(C):
    T = C.shape[0]; act = []; done = []
    for t in range(T):
        segs = runs(C[t]); used = set()
        for e in act:
            a0, b0 = e["seg"][-1]
            hit = [s for s in segs if s[0] <= b0 + 1 and s[1] >= a0 - 1]
            if hit:
                a, b = min(h[0] for h in hit), max(h[1] for h in hit)
                e["seg"].append((a, b)); e["ts"].append(t); e["gap"] = 0; used.update(hit)
            else:
                e["gap"] += 1
        for s in segs:
            if s not in used:
                act.append(dict(t0=t, seg=[s], ts=[t], gap=0))
        keep = []
        for e in act:
            (done if e["gap"] > 1 else keep).append(e)
        act = keep
    return done + act


def med(x):
    x = x[np.isfinite(x)]
    return float(np.median(x)) if len(x) else np.nan


out_all = []
for panel in (sys.argv[1:] or T2_PANELS):
    P = load(panel, "train")
    V, Q = P.V.astype(float), P.Q.astype(float)
    fin = np.isfinite(V)
    C = fin & (V <= P.vcut[None, :])
    gap = ~fin
    C[:, 1:-1] |= gap[:, 1:-1] & C[:, :-2] & C[:, 2:]
    x0 = np.concatenate([[0.0], np.cumsum(P.length)])
    n, ndays = P.n, P.T // 288
    E = [e for e in episodes(C) if len(e["ts"]) >= 3 and max(b - a + 1 for a, b in e["seg"]) >= 2]
    heads = np.array([int(np.median([b for a, b in e["seg"]])) for e in E])
    days = np.array([e["t0"] // 288 for e in E])
    sitedays = np.zeros(n)
    for h in range(n):
        sitedays[h] = len(set(days[np.abs(heads - h) <= 2]))
    f = sitedays[heads] / ndays
    rec = []
    for e, h, fe in zip(E, heads, f):
        ts = np.array(e["ts"]); seg = e["seg"]; L = len(ts)
        rt = []
        for t, (a, b) in zip(ts, seg):
            j = np.arange(b + 1, min(b + 4, n))
            if len(j) == 0:
                continue
            ff = V[t, j] > 0.8 * P.vf[j]
            if ff.sum():
                rt.append(np.median(Q[t, j[ff]]) / P.cap[j].min())
        r = med(np.array(rt))
        a0, b0 = seg[0]
        # precursor at the first head link: v/vf 10 and 30 min before the start
        t0 = ts[0]
        pre10 = V[t0 - 2, b0] / P.vf[b0] if t0 >= 2 else np.nan
        pre30 = V[t0 - 6, b0] / P.vf[b0] if t0 >= 6 else np.nan
        # tail kinematics over the first 30 min (needs >= 7 steps)
        dx = pred = np.nan
        if L >= 7:
            a6 = seg[6][0]
            dx = x0[a0] - x0[a6]                                   # km grown upstream
            ju = np.arange(max(a0 - 4, 0), max(a0 - 1, 0))
            if len(ju):
                qu, vu = med(Q[t0, ju]), med(V[t0, ju])
                jd = np.arange(b0 + 1, min(b0 + 4, n))
                qo = med(np.concatenate([Q[t, jd] for t in ts[:3]])) if len(jd) else np.nan
                if np.isfinite(qu) and np.isfinite(vu) and vu > 0 and np.isfinite(qo):
                    ku = qu / vu; kq = P.kj[a0] - qo / P.w[a0]
                    ws = (qu - qo) / (ku - kq)                     # km/h, negative = upstream
                    pred = -ws * 0.5
        # dissolution: head displacement over the last 30 min (upstream positive), km/h
        vh = np.nan
        if L >= 7:
            bL, b6 = seg[-1][1], seg[-7][1]
            vh = (x0[b6 + 1] - x0[bL + 1]) / 0.5
        vmin = med(np.array([np.nanmin(V[t, a:b + 1] / P.vf[a:b + 1]) for t, (a, b) in zip(ts, seg)]))
        jg = np.arange(b0 + 1, min(b0 + 4, n)); g = P.cap[jg].min() / P.cap[b0] if len(jg) else np.nan
        onr = float(P.on_attached[b0:min(b0 + 2, n)].any()); offr = float(P.off_attached[b0:min(b0 + 2, n)].any())
        conn = float(P.topo.iloc[b0:min(b0 + 2, n)].astype(str).apply(lambda c: c.str.contains("CONN")).any().any())
        def dfc(t):
            if t < 0: return np.nan
            ju_ = np.arange(max(b0 - 3, 0), b0); jd_ = np.arange(b0 + 1, min(b0 + 4, n))
            if len(ju_) == 0 or len(jd_) == 0: return np.nan
            return (med(Q[t, ju_]) - med(Q[t, jd_])) / P.cap[b0]
        d1, d12 = dfc(t0 - 1), dfc(t0 - 12)
        qload = Q[t0 - 1, b0] / P.cap[b0] if t0 >= 1 else np.nan
        rec.append((panel, fe, r, L * 5, max(b - a + 1 for a, b in seg), (t0 % 288) / 12, pre10, pre30, dx, pred, vh, vmin,
                    float(np.median(P.w[a0:b0 + 1])), g, onr, offr, conn, d1, d12, qload))
    R = np.array([x[1:] for x in rec], float)
    out_all.append((panel, R))
    jg_all = [P.cap[np.arange(i + 1, min(i + 4, n))].min() / P.cap[i] if i < n - 1 else np.nan for i in range(n)]
    conn_all = P.topo.astype(str).apply(lambda c: c.str.contains("CONN")).any(axis=1).to_numpy()
    print(panel, "episodes", len(E), "days", ndays, "links with g<0.95:", round(np.nanmean(np.array(jg_all) < 0.95), 3),
          "links on-ramp:", round(P.on_attached.mean(), 3), "CONN links:", round(conn_all.mean(), 3), flush=True)

cols = ["f", "r", "dur", "len", "tod", "pre10", "pre30", "dx", "pred", "vh", "vmin", "w", "g", "onr", "offr", "conn", "d1", "d12", "qload"]
R = np.concatenate([r for _, r in out_all])
ix = {c: i for i, c in enumerate(cols)}
def grp(m, name):
    S = R[m]
    def q(c, p=50):
        v = S[:, ix[c]]; v = v[np.isfinite(v)]
        return np.percentile(v, p) if len(v) else np.nan
    long = S[np.isfinite(S[:, ix["dx"]]) & np.isfinite(S[:, ix["pred"]])]
    if len(long) > 5:
        cc = np.corrcoef(long[:, ix["dx"]], long[:, ix["pred"]])[0, 1]
        mae_rh = np.mean(np.abs(long[:, ix["dx"]] - long[:, ix["pred"]])); mae_p = np.mean(np.abs(long[:, ix["dx"]]))
    else:
        cc = mae_rh = mae_p = np.nan
    rr = S[:, ix["r"]]; rr = rr[np.isfinite(rr)]
    hist = np.histogram(rr, bins=[0, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.05, 9])[0] / max(len(rr), 1)
    print(f"[{name}] n={len(S)} dur med {q('dur'):.0f} p90 {q('dur',90):.0f} min | len med {q('len'):.0f} | r med {q('r'):.3f} "
          f"p25 {q('r',25):.3f} | r hist <.5 .5-.6 .6-.7 .7-.8 .8-.9 .9-.95 .95-1.05 >1.05: {np.round(hist,3).tolist()}")
    print(f"    pre10 v/vf med {q('pre10'):.3f} p10 {q('pre10',10):.3f} | pre30 med {q('pre30'):.3f} | vmin/vf med {q('vmin'):.3f} | tod med {q('tod'):.1f} "
          f"(p10 {q('tod',10):.1f} p90 {q('tod',90):.1f})")
    print(f"    geometry at head: g(dn min cap/head cap)<0.95 {np.nanmean(S[:, ix['g']] < 0.95):.3f}, on-ramp {np.nanmean(S[:, ix['onr']]):.3f}, off-ramp {np.nanmean(S[:, ix['offr']]):.3f}, CONN {np.nanmean(S[:, ix['conn']]):.3f}")
    dd = S[:, ix['d1']] - S[:, ix['d12']]
    print(f"    flow deficit across head (up-dn)/cap: t0-5min med {q('d1'):.3f}, t0-60min med {q('d12'):.3f}, change med {np.nanmedian(dd):.3f}, share change>0.05 {np.nanmean(dd > 0.05):.3f} | head q/cap at t0-5 med {q('qload'):.3f} p90 {q('qload',90):.3f}")
    print(f"    tail 30min: n={len(long)} actual med {q('dx'):.2f} km, RH pred med {q('pred'):.2f} km, corr {cc:.3f}, MAE RH {mae_rh:.2f} vs persist {mae_p:.2f}"
          f" | head last-30min speed med {q('vh'):.1f} km/h (w med {q('w'):.1f})")

f = R[:, ix["f"]]; r = R[:, ix["r"]]
grp(f >= 0.2, "recurrent f>=0.2")
grp((f >= 0.05) & (f < 0.2), "mid 0.05<=f<0.2")
grp(f < 0.05, "rare f<0.05")
grp((f < 0.05) & (R[:, ix["dur"]] >= 30), "rare & dur>=30min")
grp((f < 0.05) & (r < 0.8), "rare & r<0.8")
grp((f >= 0.2) & (r < 0.8), "recurrent & r<0.8")
for p, Rp in out_all:
    fp = Rp[:, 0]; rp = Rp[:, 1]
    m = fp < 0.05
    print(f"  {p}: rare {m.sum()} ({m.mean():.3f}), rare&dur>=30 {(m & (Rp[:,2]>=30)).sum()}, rare r med {np.nanmedian(rp[m]):.3f}, "
          f"recurrent r med {np.nanmedian(rp[fp>=0.2]):.3f}")
np.save(r"C:/Users/user/tfb/review/m2_incident_physics.npy", R)
