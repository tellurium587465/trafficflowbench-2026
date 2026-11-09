"""Clean randomized worlds for FINAL_SELECTION (b) (pre-registered 10/13): the randomization scales are NOT fitted to
any test-month statistic. Each world draws its scales from a broad a-priori set (domain randomization):
  junction log-ratio shift  SIG_RAT in {0.2, 0.4, 0.8} (relative to each junction's mean net log ratio, as gen.py)
  upstream demand           SIG_UP  in {0.01, 0.03, 0.05}
  capacity per link         SIG_CAP 0.015 (the training months' structural uncertainty, as gen.py)
  free-flow speed           no redraw (gen.py's 1% came from the test months)
Same CTM, incident law, noise, masks and release fields as gen.py (world A not rewritten). Worlds twinC0..C<n-1>,
template month k % 9, calendars disjoint from every other family (offset 280). Writes caches like gen.py (E: via the
work/cache junction: run only when the E: HDD is free).
usage: python gen_clean.py <panel|ALL> [n=72] [seed=31]"""
import sys, time, zlib, numpy as np, pandas as pd
sys.path.insert(0, r"C:/Users/user/tfb/src"); sys.path.insert(0, r"C:/Users/user/tfb/twin")
from cachelib import load, T2_PANELS
from ctm import junction_ratios
import gen

RATS, UPS = (0.2, 0.4, 0.8), (0.01, 0.03, 0.05)


def main(panel, n=72, seed=31):
    t0 = time.time(); PT = load(panel, "train"); D = PT.T // 288
    rng = np.random.default_rng([seed, zlib.crc32(panel.encode())])
    months = sorted(set(str(x)[:7] for x in PT.day[:D]))
    base_lr = np.log(junction_ratios(PT)).mean(axis=(0, 1))
    one = np.ones(PT.n)
    for k in range(n):
        tm = months[k % len(months)]; days = [d for d in range(D) if str(PT.day[d]).startswith(tm)][:30]
        s_rat, s_up = RATS[rng.integers(3)], UPS[rng.integers(3)]
        cap_mult = np.exp(gen.SIG_CAP * rng.standard_normal(PT.n))
        rat = s_rat * np.abs(base_lr) * rng.standard_normal(PT.n - 1); upm = float(np.exp(s_up * rng.standard_normal()))
        V, Q, log = gen.simulate(PT, days, rng, cap_mult, one, rat, upm)
        y, m = gen.far_calendar(pd.Timestamp(str(PT.day[days[0]])).dayofweek, k + 280)
        base = pd.Timestamp(y, m, 1)
        times = [(base + pd.Timedelta(minutes=5 * i)).strftime("%Y-%m-%dT%H:%M:%SZ") for i in range(len(days) * 288)]
        gen.write(panel, f"twinC{k}", PT, days, V, Q, rng, times,
                  dict(world=f"C{k}", template=tm, calendar=f"{y}-{m:02d}", sig_rat=s_rat, sig_up=s_up, up_mult=upm,
                       cap_sd=float(np.log(cap_mult).std()), incidents=log, days=len(days)))
    print(panel, f"twinC0..{n-1}", f"{time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    pans = T2_PANELS if sys.argv[1] == "ALL" else sys.argv[1].split(",")
    for p in pans:
        main(p, int(sys.argv[2]) if len(sys.argv) > 2 else 72, int(sys.argv[3]) if len(sys.argv) > 3 else 31)
