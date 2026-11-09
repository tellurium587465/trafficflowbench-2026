"""Twin fidelity repair v2 (10/13): stochastic capacity per regular site. fid_replay.py showed that the measured
breakdown flow fixes WHICH site breaks (D7_I10_E same-site 0.00 -> 0.92) but a fixed threshold breaks on too many
mornings (G36: real D7 sites often carry more than their usual breakdown flow without breaking).
Per regular site s (G34 events, n >= 3; links b..b+2): capacity ~ LogNormal(mu, sigma), fitted by censored maximum
likelihood: breakdown mornings give the capacity (= the breakdown flow), mornings without a queue at s give a lower
bound (the largest 30-min mean flow carried). In the replay each day draws its own capacity per site (capf), other
links as in fid_replay 'cap'. Scored as fid_replay (same site, extra / missing queue mornings, dominance).
usage: python fid_replay2.py [panels=ALL] [day=0|1]
Review 10/13: rewritten to be out of sample with a matched control (see the comment above the main loop)."""
import sys, time, numpy as np, pandas as pd, warnings
from scipy.optimize import minimize
from scipy.stats import norm
warnings.simplefilter("ignore")
sys.argv += [] if len(sys.argv) > 1 else ["ALL"]
_args = list(sys.argv)
sys.argv = ["x", "NONE"]
exec(open(r"C:/Users/user/tfb/twin/fid_replay.py", encoding="utf-8").read().split("pans = ")[0])
sys.argv = _args


def site_dists(P):
    """{b: (mu, sigma)} by censored lognormal MLE"""
    D = P.T // 288; qv = (P.V <= P.vcut[None, :]) & np.isfinite(P.V); out = {}
    for b, g in E[E.panel == P.panel].groupby("b"):
        if len(g) < 3:
            continue
        brk = np.log(g.bf.dropna().to_numpy()); surv = []
        for d in range(D):
            lo, hi = d * 288 + 60, d * 288 + 144
            if qv[lo:hi, max(0, b - 4):b].any():
                continue
            m = np.nanmean(P.Q[lo:hi, b:b + 3].reshape(-1, 6, min(3, P.n - b)), axis=1)
            if np.isfinite(m).any():
                surv.append(np.log(np.nanmax(m)))
        surv = np.array(surv)
        def nll(th):
            mu, ls = th; s = np.exp(ls)
            return -(norm.logpdf(brk, mu, s).sum() + norm.logsf(surv, mu, s).sum())
        r = minimize(nll, [brk.mean(), np.log(max(brk.std(), 0.01))], method="Nelder-Mead")
        out[b] = (r.x[0], float(np.exp(r.x[1])), len(brk), len(surv))
    return out


def replay2(P, cap_mult, dists, per_day, seed=0, stochastic=True, jr_rows=None):
    """stochastic=False: each site at the median of its fitted capacity (exp(mu)) every day = the matched control"""
    rng = np.random.default_rng(seed)
    C = Corridor(P, cap_mult=cap_mult); D = P.T // 288
    RAT = junction_ratios(P, jr_rows)
    dayt = np.array([pd.Timestamp(str(x)).dayofweek >= 5 for x in P.day[:D]]).astype(int)
    RD = day_ratios(P, RAT) if per_day else RAT[dayt]
    CR = C.cell_ratios(RD)
    up = pd.Series(P.Q[:, 0].astype(float)).interpolate(limit_direction="both").to_numpy()[:D * 288].reshape(D, 288)
    capf = np.ones((D, 288, C.NC))
    for b, (mu, s) in dists.items():
        cells = np.unique(C.cell_of[b:b + 3])
        draw = np.exp(mu + s * rng.standard_normal(D)) if stochastic else np.full(D, np.exp(mu))
        for c in cells:
            capf[:, :, c] = np.minimum(capf[:, :, c], (draw / C.cap[c])[:, None])
    Vc, Fc = C.run(up, CR, capf)
    return Vc[:, :, C.cell_of].reshape(-1, P.n)


# review 10/13: out of sample (identify on months 0-5, score the first-queue site on months 6-8), one shared estimator
# (gen2.site_dists, symmetric statistic), and a matched deterministic control (same sites, median capacity)
import gen2
pans = T2_PANELS if sys.argv[1] == "ALL" else sys.argv[1].split(",")
pday = len(sys.argv) > 2 and sys.argv[2] == "1"
rows = []
for p in pans:
    t0 = time.time(); P = load(p, "train"); D = P.T // 288
    months = sorted(set(str(x)[:7] for x in P.day[:D])); keep = [d for d in range(D) if str(P.day[d])[:7] in months[:6]]
    test = set(str(P.day[d]) for d in range(D) if str(P.day[d])[:7] in months[6:])
    jr_rows = np.isin(np.arange(P.T) // 288, keep)
    real = {k: v for k, v in first_sites(P.V[:D * 288], P.vcut, P.E[:D * 288], P.day).items() if k in test}
    hr = pd.Series([v[1] // 3 for v in real.values()]).value_counts(normalize=True)
    dists = gen2.site_dists(P, keep); cm = gen2.base_cap_mult(P, keep, dists)
    for nm, sto in (("stochastic", True), ("median", False)):
        V = replay2(P, cm, dists, pday, stochastic=sto, jr_rows=jr_rows)
        tw = {k: v for k, v in first_sites(V, P.vcut, P.E[:D * 288], P.day).items() if k in test}
        same, nc, only_r, only_t, dom = score(real, tw)
        rows.append(dict(panel=p, variant=nm, same_site=same, common=nc, only_real=only_r, only_twin=only_t, dom_twin=dom,
                         dom_real=hr.iloc[0] if len(hr) else np.nan, sites=len(dists)))
        print(p, nm, f"same {same:.3f} common {nc} only_real {only_r} only_twin {only_t} dom {dom:.3f} sites {len(dists)} {time.time()-t0:.0f}s", flush=True)
R = pd.DataFrame(rows)
print(R.pivot(index="panel", columns="variant", values="same_site").round(3).to_string())
print(R.groupby("variant")[["same_site", "only_real", "only_twin", "dom_twin", "dom_real"]].mean().round(3).to_string())
