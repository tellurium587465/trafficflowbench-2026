"""Digital twin stage 2: synthetic months as panel caches (work/cache/<panel>_<split>.npz), readable by cachelib.load.

World A ('twinA', the training scenario): every real train day replayed with its own upstream demand, the measured
junction ratios and fd_parameters; random incidents. Test worlds ('twinT<k>', one 30-day month each, a new scenario
seed each): capacity x exp(N(0, SIG_CAP)) per link, free speed x exp(N(0, SIG_VF)) per link (F3: ~1% in the test
months), junction log-ratios + N(0, SIG_RAT) per junction, upstream demand x exp(N(0, SIG_UP)); same incident law.
Days of a test month copy the upstream demand, masks, eligibility and ramps of a real train month (template), and get
a synthetic calendar in a far year whose 1st has the template's weekday (no clash with real months; prior days in the
month exist exactly as in the test months). Observations = simulated state x (1 + noise) (speed 0.8%, flow 2.3%,
GENERATOR.md), NaN where the template is NaN; blank / target / regime / E / P / O / ramps copied from the template.

The world parameters are written to work/twin/<panel>_<split>.json. Nothing from the validation/private months is used.
usage: python gen.py <panel|ALL> [n_months=8] [seed=0] [prefix=T|R|A6|U|S] [template month indices, e.g. 0,1,2,3,4,5]   (prefix R = randomized TRAINING worlds, another seed;
       never the test worlds)
"""
import sys, json, time, calendar, zlib, numpy as np, pandas as pd
from pathlib import Path
sys.path.insert(0, r"C:/Users/user/tfb/src"); sys.path.insert(0, r"C:/Users/user/tfb/twin")
from cachelib import load, CACHE, T2_PANELS
from ctm import Corridor, junction_ratios

SIG_CAP, SIG_VF, SIG_RAT, SIG_UP = 0.015, 0.01, 0.4, 0.03           # SIG_RAT: relative to each junction's own net flow
# (v2, 10/12: a flat 0.01 per junction compounded along 100-350 junctions to 10-19% at the downstream end, and the
#  test worlds moved the congestion map by TV up to 1.0 vs the real 0.10-0.40; real month-to-month link flows move 1-6%)
INC_PER_MONTH, INC_FACTOR, INC_DUR = 0.8, 0.5, (6, 18)           # per panel-month, capacity factor, 5-min rows
NOISE_V, NOISE_Q = 0.008, 0.023
OUT = Path(r"C:/Users/user/tfb/work/twin"); OUT.mkdir(parents=True, exist_ok=True)


def incidents(rng, nd, NC):
    capf = np.ones((nd, 288, NC), np.float32); log = []
    for _ in range(rng.poisson(INC_PER_MONTH * nd / 30)):
        d = rng.integers(nd); r0 = rng.integers(72, 240); dur = rng.integers(*INC_DUR); c = rng.integers(NC)
        capf[d, r0:r0 + dur, c] = INC_FACTOR; log.append((int(d), int(r0), int(dur), int(c)))
    return capf, log


def simulate(PT, days, rng, cap_mult, vf_mult, rat_shift, up_mult):
    """days: template day indices of PT. Returns V, Q on links (len(days)*288, n), incident log."""
    vf0 = PT.vf.copy(); PT.vf = PT.vf * vf_mult
    try:
        C = Corridor(PT, cap_mult=cap_mult)
    finally:
        PT.vf = vf0
    RAT = junction_ratios(PT) * np.exp(rat_shift)[None, None, :]
    dayt = np.array([pd.Timestamp(str(PT.day[d])).dayofweek >= 5 for d in days]).astype(int)
    CR = C.cell_ratios(RAT)[dayt]                                              # (nd, 288, NC)
    up = pd.Series(PT.Q[:, 0].astype(float)).interpolate(limit_direction="both").to_numpy()
    U = np.stack([up[d * 288:(d + 1) * 288] for d in days]) * up_mult
    capf, log = incidents(rng, len(days), C.NC)
    Vc, Fc = C.run(U, CR, capf)
    # inside a cell the flow is the one entering it (its junction ratios act at the exit), so link i carries the cell
    # flow x the product of the ratios of the junctions m < i that belong to the same cell
    lr = np.log(RAT)                                                           # (2, 288, n-1)
    same = C.cell_of[:-1] == C.cell_of[1:]                                     # junction m inside a cell (m -> m+1)
    G = np.zeros((2, 288, PT.n))
    for i in range(1, PT.n):
        G[:, :, i] = np.where(same[i - 1], G[:, :, i - 1] + lr[:, :, i - 1], 0.0)
    G = np.exp(G)[dayt]                                                        # (nd, 288, n)
    V = Vc[:, :, C.cell_of].reshape(-1, PT.n); Q = (Fc[:, :, C.cell_of] * G).reshape(-1, PT.n)
    return V.astype(np.float32), Q.astype(np.float32), log


def far_calendar(first_dow, k):
    """the k-th month (k>=0) after 2040-01 whose 1st is on weekday first_dow and that has >= 30 days."""
    y, m, found = 2040, 1, -1
    while True:
        if pd.Timestamp(y, m, 1).dayofweek == first_dow and calendar.monthrange(y, m)[1] >= 30:
            found += 1
            if found == k:
                return y, m
        m += 1
        if m > 12:
            y, m = y + 1, 1


def write(panel, split, PT, days, V, Q, rng, times, meta):
    rows = np.concatenate([np.arange(d * 288, (d + 1) * 288) for d in days])
    Vo = V * (1 + NOISE_V * rng.standard_normal(V.shape)).astype(np.float32)
    Qo = Q * (1 + NOISE_Q * rng.standard_normal(Q.shape)).astype(np.float32)
    nan = ~np.isfinite(PT.V[rows]); Vo[nan] = np.nan; Qo[nan] = np.nan
    np.savez(CACHE / f"{panel}_{split}.npz", times=np.array(times, dtype=object), links=PT.links, V=Vo, Q=Qo,
             O=PT.O[rows], P=PT.P[rows], E=PT.E[rows], regime=PT.regime[rows], target=PT.target[rows], blank=PT.blank[rows],
             R=PT.R[rows], RE=PT.RE[rows], ramp_ids=PT.ramp_ids, ramp_type=PT.ramp_type, ramp_link=PT.ramp_link)
    json.dump(meta, open(OUT / f"{panel}_{split}.json", "w"), indent=1, default=float)


def main(panel, n_test, seed, prefix="T", tmonths=None):
    t0 = time.time(); PT = load(panel, "train"); D = PT.T // 288
    rng = np.random.default_rng([seed, zlib.crc32(panel.encode())])   # stable across processes (hash() is salted)
    months = sorted(set(str(x)[:7] for x in PT.day[:D]))
    if tmonths is not None:                                                    # template months restricted (disjoint-day tests)
        months = [months[i] for i in tmonths]
    if prefix in ("T", "A6"):                                                  # world A (or A6: its template months only), no redraw
        one = np.ones(PT.n); zero = np.zeros(PT.n - 1)
        adays = [d for d in range(D) if str(PT.day[d])[:7] in months] if prefix == "A6" else list(range(D))
        V, Q, log = simulate(PT, adays, rng, one, one, zero, 1.0)
        rows = np.concatenate([np.arange(d * 288, (d + 1) * 288) for d in adays])
        nameA = "twinA" if prefix == "T" else "twinA6"
        write(panel, nameA, PT, adays, V, Q, rng, list(PT.times[rows]), dict(world=nameA[4:], incidents=log, days=len(adays)))
        print(panel, nameA, f"{time.time()-t0:.0f}s", flush=True)
        if prefix == "A6":
            return
    for k in range(n_test):
        tm = months[k % len(months)]; days = [d for d in range(D) if str(PT.day[d]).startswith(tm)][:30]
        cap_mult = np.exp(SIG_CAP * rng.standard_normal(PT.n)); vf_mult = np.exp(SIG_VF * rng.standard_normal(PT.n))
        base_lr = np.log(junction_ratios(PT)).mean(axis=(0, 1))                 # each junction's mean net log ratio
        rat = SIG_RAT * np.abs(base_lr) * rng.standard_normal(PT.n - 1); upm = float(np.exp(SIG_UP * rng.standard_normal()))
        V, Q, log = simulate(PT, days, rng, cap_mult, vf_mult, rat, upm)
        y, m = far_calendar(pd.Timestamp(str(PT.day[days[0]])).dayofweek, k + {"T": 0, "R": 40, "U": 80, "S": 120}[prefix])   # disjoint calendars per world family
        base = pd.Timestamp(y, m, 1)
        times = [(base + pd.Timedelta(minutes=5 * i)).strftime("%Y-%m-%dT%H:%M:%SZ") for i in range(len(days) * 288)]
        write(panel, f"twin{prefix}{k}", PT, days, V, Q, rng, times,
              dict(world=f"{prefix}{k}", template=tm, calendar=f"{y}-{m:02d}", up_mult=upm, cap_sd=float(np.log(cap_mult).std()),
                   incidents=log, days=len(days)))
        print(panel, f"twin{prefix}{k}", tm, f"-> {y}-{m:02d}", f"{time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    pans = T2_PANELS if sys.argv[1] == "ALL" else sys.argv[1].split(",")
    for p in pans:
        main(p, int(sys.argv[2]) if len(sys.argv) > 2 else 8, int(sys.argv[3]) if len(sys.argv) > 3 else 0,
             sys.argv[4] if len(sys.argv) > 4 else "T", [int(x) for x in sys.argv[5].split(",")] if len(sys.argv) > 5 else None)
