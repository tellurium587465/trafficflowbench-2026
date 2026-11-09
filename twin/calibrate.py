"""Twin calibration (system identification on TRAIN months only): per-link capacity multipliers so that world A's
queue heads (bottlenecks) occur where the real ones do. F57: mixing twin worlds into the real training hurt the real
onset (-0.024) because the twin's 'which bottleneck activates at which flow' differs from reality; this narrows it.

Rule (fixed point): head frequency per link (downstream-most queued link of each run, daytime rows), twin vs real;
the bottleneck of a head at link i is link i+1; log cap_mult[i+1] += ETA * (H_twin[i] - H_real[i]) / max(H), ETA 0.02
(v1 used a log ratio and diverged: links never a real head got huge steps), clipped to [0.90, 1.15]. Fit days: a sample of months 0-5; held-out check: months 6-8 (not used for fitting), map TV
(per-link queued share, normalised) and head-site agreement, before and after.
Output: work/twin/<panel>_capmult.npy. usage: python calibrate.py <panel|ALL> [iters=8]
"""
import sys, time, numpy as np, pandas as pd
sys.path.insert(0, r"C:/Users/user/tfb/src"); sys.path.insert(0, r"C:/Users/user/tfb/twin")
from cachelib import load, T2_PANELS
from ctm import Corridor, junction_ratios

ETA, EPS, LO, HI = 0.02, 2e-4, 0.90, 1.15                         # <= 2% capacity change per iteration
OUT = "C:/Users/user/tfb/work/twin/"


def sim(PT, days, cap_mult, RAT, up):
    C = Corridor(PT, cap_mult=cap_mult)
    dayt = np.array([pd.Timestamp(str(PT.day[d])).dayofweek >= 5 for d in days]).astype(int)
    CR = C.cell_ratios(RAT)[dayt]; U = np.stack([up[d * 288:(d + 1) * 288] for d in days])
    Vc, _ = C.run(U, CR, None)
    return Vc[:, :, C.cell_of].reshape(-1, PT.n)


def heads(Q, rows_day):
    """head frequency per link over daytime rows (06-21 h)"""
    n = Q.shape[1]; h = np.zeros(n); tod = rows_day % 288; day = (tod >= 72) & (tod < 252)
    for r in np.flatnonzero(day):
        x = Q[r]
        if x.any():
            idx = np.flatnonzero(x); ends = idx[np.r_[np.diff(idx) > 1, True]]; h[ends] += 1
    return h / max(day.sum(), 1)


def tv(a, b):
    a = a / max(a.sum(), 1e-12); b = b / max(b.sum(), 1e-12); return 0.5 * np.abs(a - b).sum()


def main(panel, iters):
    t0 = time.time(); PT = load(panel, "train"); D = PT.T // 288
    mon = np.array([str(x)[:7] for x in PT.day[:D]]); months = sorted(set(mon))
    fit_m, hold_m = months[:6], months[6:]
    rng = np.random.default_rng(0)
    fit_days = sorted(rng.choice([d for d in range(D) if mon[d] in fit_m], 60, replace=False))
    hold_days = [d for d in range(D) if mon[d] in hold_m]
    RAT = junction_ratios(PT, rows=np.isin(np.repeat(mon, 288)[:PT.T], fit_m))       # ratios from fit months only
    up = pd.Series(PT.Q[:, 0].astype(float)).interpolate(limit_direction="both").to_numpy()
    Qr = (np.nan_to_num(PT.V, nan=1e9) <= PT.vcut[None, :]) & PT.E
    rows = lambda ds: np.concatenate([np.arange(d * 288, (d + 1) * 288) for d in ds])
    Hr_fit = heads(Qr[rows(fit_days)], rows(fit_days)); Hr_hold = heads(Qr[rows(hold_days)], rows(hold_days))
    qr_hold = Qr[rows(hold_days)].mean(0)

    def evaluate(cm, days, Hr, qr=None):
        V = sim(PT, days, cm, RAT, up); E = PT.E[rows(days)]
        Qs = (V <= PT.vcut[None, :]) & E
        H = heads(Qs, rows(days)); out = dict(head_tv=tv(H, Hr))
        if qr is not None:
            out["map_tv"] = tv(Qs.mean(0), qr)
        return H, out

    cm = np.ones(PT.n)
    _, before = evaluate(cm, hold_days, Hr_hold, qr_hold)
    for it in range(iters):
        H, fit = evaluate(cm, fit_days, Hr_fit)
        g = (H - Hr_fit) / max(H.max(), Hr_fit.max(), 1e-9)                       # bounded in [-1, 1]
        step = np.zeros(PT.n); step[1:] = g[:-1]                                   # head at i -> bottleneck i+1
        cm = np.clip(cm * np.exp(ETA * step), LO, HI)
        print(f"  {panel} iter {it}: fit head TV {fit['head_tv']:.3f}  cap_mult range {cm.min():.3f}-{cm.max():.3f}", flush=True)
    _, after = evaluate(cm, hold_days, Hr_hold, qr_hold)
    np.save(OUT + f"{panel}_capmult.npy", cm)
    print(f"{panel}: HELD-OUT months {hold_m[0]}..{hold_m[-1]}: head TV {before['head_tv']:.3f} -> {after['head_tv']:.3f}, "
          f"map TV {before['map_tv']:.3f} -> {after['map_tv']:.3f}  ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    for p in (T2_PANELS if sys.argv[1] == "ALL" else sys.argv[1].split(",")):
        main(p, int(sys.argv[2]) if len(sys.argv) > 2 else 8)
