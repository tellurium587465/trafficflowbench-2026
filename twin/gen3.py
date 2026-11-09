"""Digital twin v3 (10/13): gen2.py (stochastic site capacities, identification on months 0-5) with the scenario redraw
re-fitted to the real one (shift_law.py, validation month, distribution shape only). Real redraws move most connectors
a lot (share of junctions moving > 3 within-month sd: 9% / 46% / 82% by connector size, |delta log ratio| median
0.006 / 0.010 / 0.012) while gen.py moved every junction by 40% of its own size (big connectors about half the real
amount, small ones almost not at all). Upstream demand: sd 0.015 (real changes 0.8-1.9%).
  twinL<k> : template month k % 6, 30 days; per junction: with the bin's moving probability a shift of random sign and a
             magnitude drawn from the bin's moving |delta| quantiles, else N(0, 0.0015); capacity / free speed per link
             as gen.py; per-site capacity median shift N(0, 0.03) and per-day draws as gen2.py.
usage: python gen3.py <panel|ALL> [n=16] [seed=11] [prefix=L|M]   (M = held-out test redraws, another seed and calendar)"""
import sys, json, time, zlib, numpy as np, pandas as pd
sys.path.insert(0, r"C:/Users/user/tfb/src"); sys.path.insert(0, r"C:/Users/user/tfb/twin")
from cachelib import load, T2_PANELS
from ctm import junction_ratios
import gen, gen2
LAW = json.load(open(r"C:/Users/user/tfb/work/twin/shift_law.json"))
SIG_UP3, REST = 0.015, 0.0015


def law_size(PT):
    """connector size exactly as shift_law.py defines it (review 10/13): |mean over the train months of the monthly
    median free-flow log ratio on weekday mornings 06:00-10:00|"""
    D = PT.T // 288; months = sorted(set(str(x)[:7] for x in PT.day[:D])); M = []
    for m in months:
        days = [d for d in range(D) if pd.Timestamp(str(PT.day[d])).dayofweek < 5 and str(PT.day[d])[:7] == m]
        rows = np.concatenate([np.arange(d * 288 + 72, d * 288 + 120) for d in days])
        V, Q = PT.V[rows], PT.Q[rows]
        ff = np.isfinite(V) & np.isfinite(Q) & (V > 0.85 * PT.vf[None, :]) & (Q > 100); ok = ff[:, :-1] & ff[:, 1:]
        M.append(np.nanmedian(np.where(ok, np.log(np.maximum(Q[:, 1:], 1) / np.maximum(Q[:, :-1], 1)), np.nan), axis=0))
    return np.nan_to_num(np.abs(np.nanmean(np.array(M), axis=0)))


def law_shift(size, rng):
    out = np.zeros(len(size))
    for j, s in enumerate(size):
        bins = sorted(LAW.values(), key=lambda v: v["lo"])                    # review 10/13: clamp to the first / last bin
        b = next((v for v in bins if v["lo"] < s <= v["hi"]), bins[0] if s <= bins[0]["hi"] else bins[-1])
        if rng.random() < b["share_moving"]:
            qs = np.array(b["moving_abs_q"]); u = rng.random()
            mag = np.interp(u, [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0], np.concatenate([[qs[0] * 0.8], qs, [qs[-1] * 1.3]]))
            out[j] = mag * (1 if rng.random() < b["sign_pos"] else -1)
        else:
            out[j] = REST * rng.standard_normal()
    return out


def main(panel, n=16, seed=11, prefix="L"):
    t0 = time.time(); PT = load(panel, "train"); D = PT.T // 288
    rng = np.random.default_rng([seed, zlib.crc32(panel.encode())])
    months = gen2.months_of(PT)[:6]
    keep = [d for d in range(D) if str(PT.day[d])[:7] in months]
    dists = gen2.site_dists(PT, keep); cm = gen2.base_cap_mult(PT, keep, dists)
    gen2.JR_ROWS = np.isin(np.arange(PT.T) // 288, keep)                     # junction ratios from months 0-5 only
    size = law_size(PT)
    for k in range(n):
        tm = months[k % len(months)]; days = [d for d in range(D) if str(PT.day[d]).startswith(tm)][:30]
        cap_mult = cm * np.exp(gen.SIG_CAP * rng.standard_normal(PT.n)); vf_mult = np.exp(gen.SIG_VF * rng.standard_normal(PT.n))
        rat = law_shift(size, rng); upm = float(np.exp(SIG_UP3 * rng.standard_normal()))
        shift = {b: float(gen2.SIG_SITE * rng.standard_normal()) for b in dists}
        V, Q, log = gen2.simulate2(PT, days, rng, cap_mult, vf_mult, rat, upm, dists, shift)
        y, m = gen.far_calendar(pd.Timestamp(str(PT.day[days[0]])).dayofweek, k + {"L": 200, "M": 240}[prefix])
        base = pd.Timestamp(y, m, 1)
        times = [(base + pd.Timedelta(minutes=5 * i)).strftime("%Y-%m-%dT%H:%M:%SZ") for i in range(len(days) * 288)]
        gen.write(panel, f"twin{prefix}{k}", PT, days, V, Q, rng, times, dict(world=f"{prefix}{k}", template=tm, up_mult=upm, incidents=log, days=len(days),
                                                                         moving_share=float(np.mean(np.abs(rat) > 3 * REST))))
    print(panel, f"twin{prefix}0..{n-1}", f"{time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    pans = T2_PANELS if sys.argv[1] == "ALL" else sys.argv[1].split(",")
    for p in pans:
        main(p, int(sys.argv[2]) if len(sys.argv) > 2 else 16, int(sys.argv[3]) if len(sys.argv) > 3 else 11, sys.argv[4] if len(sys.argv) > 4 else "L")
