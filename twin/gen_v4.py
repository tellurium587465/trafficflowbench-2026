"""Twin v4 worlds (10/14, TWIN_V3_PLAN.md; fidelity in twin3_check.py): the clean C-world randomization of gen_clean.py
(no test-month statistic: junction log-ratio shift SIG_RAT in {0.2, 0.4, 0.8}, upstream demand SIG_UP in {0.01, 0.03,
0.05}, incidents and noise as gen.py) on the v4 simulator:
  cells     = co-congestion blocks of the train months (onset_shape3_lib.blocks)
  bottleneck= per block site h (head block of real morning onsets, >= 5): a per-day threshold drawn from its censored
              lognormal (twin3_check.block_sites) caps the next block downstream (h+1) for the day
  recording = congested cell -> its measured discharge flow Cd (median flow of the block's congested cells) and the
              congested-branch speed at Cd; free cell -> free speed (the cliff onsets and fixed levels of the real data)
  base capacity x gen2.base_cap_mult (no spurious bottlenecks), x exp(N(0, SIG_CAP)) per link as gen.py
Structural parameters (blocks, sites, thresholds, Cd, junction ratios) come from all train months; each world copies a
template month (k % 9) like gen_clean, worlds twinV<k>, calendars offset 380 (disjoint from every other family).
usage: python gen_v4.py <panel|ALL> [n=72] [seed=41] [--dry]   (--dry: simulate 2 worlds in memory, print stats, write
       nothing; the real run writes caches on E: via work/cache - run only when the E: HDD is free)"""
import sys, time, zlib, numpy as np, pandas as pd
sys.path.insert(0, r"C:/Users/user/tfb/src"); sys.path.insert(0, r"C:/Users/user/tfb/twin")
from cachelib import load, T2_PANELS
from ctm import junction_ratios
from onset_shape3_lib import blocks, raw_matrices
import gen, gen2
from twin3_check import Corridor3, block_sites

RATS, UPS = (0.2, 0.4, 0.8), (0.01, 0.03, 0.05)


def structure(PT):
    """blocks, block sites, measured discharge per block, base capacity multipliers (all train months)"""
    D = PT.T // 288; keep = list(range(D)); rows = np.ones(PT.T, bool)
    bid = blocks(PT.V, PT, rows); bs = block_sites(PT, keep, bid)
    dists = gen2.site_dists(PT, keep); cm = gen2.base_cap_mult(PT, keep, dists)
    cg = (PT.V <= PT.vcut[None, :]) & np.isfinite(PT.Q); Cd = np.full(bid.max() + 1, np.nan)
    for j in range(bid.max() + 1):
        m = cg[:, bid == j]
        if m.sum() >= 30:
            Cd[j] = np.median(PT.Q[:, bid == j][m])
    return bid, bs, cm, Cd


def simulate_v4(PT, days, rng, S, rat_shift, up_mult, cap_mult):
    bid, bs, cm, Cd = S
    C = Corridor3(PT, cap_mult=cm * cap_mult, cell_of=bid)
    RAT = junction_ratios(PT) * np.exp(rat_shift)[None, None, :]
    dayt = np.array([pd.Timestamp(str(PT.day[d])).dayofweek >= 5 for d in days]).astype(int)
    CR = C.cell_ratios(RAT)[dayt]
    up = pd.Series(PT.Q[:, 0].astype(float)).interpolate(limit_direction="both").to_numpy()
    U = np.stack([up[d * 288:(d + 1) * 288] for d in days]) * up_mult
    capf, log = gen.incidents(rng, len(days), C.NC)
    cellb = {j: int(C.cell_of[np.flatnonzero(bid == j)[0]]) for j in range(bid.max() + 1)}
    for h, (m_, s_) in bs.items():
        draw = np.exp(m_ + s_ * rng.standard_normal(len(days))); c = cellb[h + 1]
        capf[:, :, c] = np.minimum(capf[:, :, c], (draw / C.cap[c])[:, None])
    Vc, Fc = C.run3(U, CR, capf, branch=True, Cd=Cd)          # cells == blocks, so Cd is indexed by cell
    lr = np.log(RAT); same = C.cell_of[:-1] == C.cell_of[1:]
    G = np.zeros((2, 288, PT.n))
    for i in range(1, PT.n):
        G[:, :, i] = np.where(same[i - 1], G[:, :, i - 1] + lr[:, :, i - 1], 0.0)
    G = np.exp(G)[dayt]
    V = Vc[:, :, C.cell_of].reshape(-1, PT.n); Q = (Fc[:, :, C.cell_of] * G).reshape(-1, PT.n)
    return V.astype(np.float32), Q.astype(np.float32), log


def main(panel, n=72, seed=41, dry=False):
    t0 = time.time(); PT = load(panel, "train"); D = PT.T // 288
    if dry:                                                    # no E: reads in a dry run: the raw C: data instead
        PT.V, PT.Q, PT.E = raw_matrices(PT)
    rng = np.random.default_rng([seed, zlib.crc32(panel.encode())])
    months = sorted(set(str(x)[:7] for x in PT.day[:D]))
    base_lr = np.log(junction_ratios(PT)).mean(axis=(0, 1)); S = structure(PT)
    print(panel, "blocks", S[0].max() + 1, "block sites", {h: round(float(np.exp(m_))) for h, (m_, s_) in S[1].items()}, flush=True)
    for k in range(2 if dry else n):
        tm = months[k % len(months)]; days = [d for d in range(D) if str(PT.day[d]).startswith(tm)][:30]
        s_rat, s_up = RATS[rng.integers(3)], UPS[rng.integers(3)]
        cap_mult = np.exp(gen.SIG_CAP * rng.standard_normal(PT.n))
        rat = s_rat * np.abs(base_lr) * rng.standard_normal(PT.n - 1); upm = float(np.exp(s_up * rng.standard_normal()))
        V, Q, log = simulate_v4(PT, days, rng, S, rat, upm, cap_mult)
        if dry:
            cg = V <= PT.vcut[None, :]
            print(f"  world {k} ({tm}, rat {s_rat}, up {s_up}): congested share {cg.mean():.4f}, morning congested share "
                  f"{cg.reshape(len(days), 288, -1)[:, 72:144].mean():.4f}, real template {(PT.V[np.concatenate([np.arange(d * 288, (d + 1) * 288) for d in days])] <= PT.vcut[None, :]).mean():.4f}, {time.time() - t0:.0f}s", flush=True)
            continue
        y, m = gen.far_calendar(pd.Timestamp(str(PT.day[days[0]])).dayofweek, k + 380)
        base = pd.Timestamp(y, m, 1)
        times = [(base + pd.Timedelta(minutes=5 * i)).strftime("%Y-%m-%dT%H:%M:%SZ") for i in range(len(days) * 288)]
        gen.write(panel, f"twinV{k}", PT, days, V, Q, rng, times,
                  dict(world=f"V{k}", template=tm, calendar=f"{y}-{m:02d}", sig_rat=s_rat, sig_up=s_up, up_mult=upm,
                       cap_sd=float(np.log(cap_mult).std()), incidents=log, days=len(days), simulator="v4"))
    print(panel, "twinV" if not dry else "dry", f"{time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    dry = "--dry" in sys.argv; args = [a for a in sys.argv[1:] if a != "--dry"]
    pans = T2_PANELS if args[0] == "ALL" else args[0].split(",")
    for p in pans:
        main(p, int(args[1]) if len(args) > 1 else 72, int(args[2]) if len(args) > 2 else 41, dry)
