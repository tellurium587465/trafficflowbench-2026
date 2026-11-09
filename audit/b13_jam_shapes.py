"""B13 (10/29, strategy lane 11, orientation): the shape of the generator's jams in the training months (truth, all 9 months).
Larmuseau #9 says a jam has a fixed head and a straight tail. Before any rule test, measure it:
congested cell = V <= vcut (0.6 vf); a masked cell takes the state of the nearest observed row within +-2 rows of the same
link when both sides agree; a still-unknown cell inside a row bridges two congested cells. Runs per row; a run continues the
event whose head (the downstream end = the higher link index, audit/a12) is within +-1 link of its own head at the previous row.
Per event (>= 6 rows): head fixedness, extent L(t) in km (head end to tail start), growth / recession slopes (least squares
from the start to the peak and from the peak to the end) and their R^2, duration, peak extent, start time of day.
Writes work/b13_events.parquet; prints medians overall and the within-site spread (by panel x head link).
usage: py b13_jam_shapes.py [panels...]"""
import sys, numpy as np, pandas as pd
sys.path.insert(0, r"C:/Users/user/tfb/src")
from cachelib import load
from common import PANELS, WORK


def fill(C):
    """C: float (T, n) with 1 congested / 0 free / nan masked -> bool congested, nan resolved in time then bridged in space"""
    D = C.copy()
    for s in (1, 2):
        up, dn = np.full_like(C, np.nan), np.full_like(C, np.nan)
        up[s:], dn[:-s] = C[:-s], C[s:]
        m = np.isnan(D) & (up == dn) & ~np.isnan(up)
        D[m] = up[m]
    B = D == 1
    for t in np.flatnonzero(np.isnan(D).any(1)):
        r = B[t]; idx = np.flatnonzero(r)
        if len(idx) < 2:
            continue
        for a, b in zip(idx[:-1], idx[1:]):
            if b - a > 1 and np.isnan(D[t, a + 1:b]).all():
                r[a + 1:b] = True
    return B


def runs(v):
    dd = np.diff(np.r_[0, v.astype(np.int8), 0]); return list(zip(np.flatnonzero(dd == 1), np.flatnonzero(dd == -1) - 1))


def slope(t, L):
    if len(t) < 3 or np.ptp(t) == 0:
        return np.nan, np.nan
    A = np.column_stack([t, np.ones_like(t)]); c, res, *_ = np.linalg.lstsq(A, L, rcond=None)
    ss = ((L - L.mean()) ** 2).sum(); r2 = 1 - ((A @ c - L) ** 2).sum() / ss if ss > 0 else np.nan
    return c[0], r2


if __name__ == "__main__":
    rows = []
    for p in (sys.argv[1:] or PANELS):
        P = load(p, "train"); C = np.where(np.isnan(P.V), np.nan, (P.V <= P.vcut[None, :]).astype(float)); B = fill(C)
        x0 = np.concatenate([[0.0], np.cumsum(P.length[:-1])]); x1 = x0 + P.length
        active = {}                                   # head -> event dict
        done = []
        for t in range(P.T):
            new = {}
            for a, b in runs(B[t]):
                prev = next((h for h in (b, b - 1, b + 1) if h in active and h not in [e["hid"] for e in new.values()]), None)
                ev = active.pop(prev) if prev is not None else {"t0": t, "heads": [], "tails": [], "hid": None}
                ev["heads"].append(b); ev["tails"].append(a); ev["hid"] = b; new[b] = ev
            done += list(active.values()); active = new
        done += list(active.values())
        for ev in done:
            n = len(ev["heads"])
            if n < 6:
                continue
            h = np.array(ev["heads"]); tl = np.array(ev["tails"]); hm = np.bincount(h).argmax()
            L = x1[h] - x0[tl]; tt = np.arange(n, dtype=float); k = int(np.argmax(L))
            gs, gr = slope(tt[:k + 1], L[:k + 1]); rs, rr = slope(tt[k:], L[k:])
            t0 = ev["t0"]; tod = int(P.times[t0][11:13]) * 12 + int(P.times[t0][14:16]) // 5
            rows.append(dict(panel=p, head=int(hm), head_fixed=float((h == hm).mean()), head_pm1=float((np.abs(h - hm) <= 1).mean()),
                             dur=n, t0=t0, tod=tod, day=P.times[t0][:10], L0=float(L[0]), Lpk=float(L[k]), tpk=k,
                             grow=gs, grow_r2=gr, rec=rs, rec_r2=rr, Lend=float(L[-1]), plateau=float((L >= 0.9 * L[k]).mean()),
                             L1=float(L[1]), L3=float(L[3]), L5=float(L[5]), tail0=int(tl[0]), head0=int(h[0]), tailpk=int(tl[k]),
                             raw0=bool(np.isfinite(P.V[t0, max(0, tl[0] - 1):h[0] + 2]).all())))   # birth row seen without fill
        print(p, "events >= 30 min:", sum(r["panel"] == p for r in rows), flush=True)
        load.cache_clear()

    D = pd.DataFrame(rows); D.to_parquet(WORK + "/b13_events.parquet", index=False)
    print(f"\nevents {len(D)} | head exactly fixed (median share of rows) {D.head_fixed.median():.3f}, within +-1 link {D.head_pm1.median():.3f}")
    print(f"duration rows median {D.dur.median():.0f} (IQR {D.dur.quantile(.25):.0f}-{D.dur.quantile(.75):.0f}) | first-row extent L0 km median {D.L0.median():.2f} | peak km {D.Lpk.median():.2f}")
    print(f"growth km/row median {D.grow.median():.3f} (R^2 median {D.grow_r2.median():.3f}) | recession km/row {D.rec.median():.3f} (R^2 {D.rec_r2.median():.3f})")
    print("peak at share of life (median)", float((D.tpk / D.dur).median()).__round__(3), "| last-row extent / peak (median)",
          float((D.Lend / D.Lpk).median()).__round__(3), "| share of life at >= 0.9 peak", float(D.plateau.median()).__round__(3))
    print("extent in the first rows (median km): L0 %.2f L1 %.2f L3 %.2f L5 %.2f" % tuple(D[c].median() for c in ("L0", "L1", "L3", "L5")))
    G = D[D.dur >= 12].groupby(["panel", "head"])
    S = G.agg(n=("dur", "size"), grow_md=("grow", "median"), grow_iqr=("grow", lambda s: s.quantile(.75) - s.quantile(.25)),
              dur_md=("dur", "median"), dur_iqr=("dur", lambda s: s.quantile(.75) - s.quantile(.25)),
              L0_md=("L0", "median"), L0_iqr=("L0", lambda s: s.quantile(.75) - s.quantile(.25)),
              tod_md=("tod", "median"), tod_iqr=("tod", lambda s: s.quantile(.75) - s.quantile(.25)),
              L0_mode_share=("L0", lambda s: (np.abs(s - s.median()) < 0.01).mean()))
    print("\nsites with >= 30 events of >= 1 h (within-site spread: IQR)")
    print(S[S.n >= 30].sort_values("n", ascending=False).head(30).round(3).to_string())
    A = D.merge(S.reset_index()[["panel", "head", "L0_md"]], on=["panel", "head"], how="left")
    print("\nall events (>= 30 min): share whose birth extent is within 10 m of its site's median: %.3f" % (np.abs(A.L0 - A.L0_md) < 0.01).mean())
