"""G84 (INSTRUMENT_V3_SPEC §2.5 / G5, 10/23): share of evidence-slot rows hidden by the selector-simulated blanks for a
given number of windows per condition the replica may pick (env TFB_M3_SIMN), on the M3 v2 windows (V11's sim part);
the official March windows' real share is 0.216 (V11).
usage: TFB_M3_SIMN=<n> python g84_sim_fidelity.py"""
import sys, os
sys.path.insert(0, r"C:/Users/user/tfb/src"); sys.argv = [sys.argv[0], "0"]
import numpy as np
import t2_m3_gate as G, m3v2
sim = []
for w in G.WID:
    m = G.META[w]
    for k in (1, 2, 3):
        if k > m["day"]:
            continue
        hid = m3v2.sim_blank_rows(m["panel"], m["day"], k, m["t"])
        for back in range(1, k + 1):
            dd = m["t"] - back * 288
            sim.append(np.mean([r_ in hid for r_ in range(dd - 6, dd + 12)]))
print(f"SIMN {m3v2.SIMN}: simulated share {np.mean(sim):.3f} (n {len(sim)}); official real 0.216")
# effective hidden share on the same definition as the official side (V11 'real'): rows with nothing observed in the
# masked validation release OR hidden by the simulated blanks
eff, nat = [], []
for w in G.WID:
    m = G.META[w]; P, _, _ = m3v2._panel(m["panel"]); obs_any = np.isfinite(P.V).any(axis=1)
    for k in (1, 2, 3):
        if k > m["day"]:
            continue
        hid = m3v2.sim_blank_rows(m["panel"], m["day"], k, m["t"])
        for back in range(1, k + 1):
            dd = m["t"] - back * 288; rr = range(dd - 6, dd + 12)
            nat.append(np.mean([not obs_any[r_] for r_ in rr])); eff.append(np.mean([(not obs_any[r_]) or (r_ in hid) for r_ in rr]))
print(f"M3 evidence slots: rows with nothing observed in the release {np.mean(nat):.3f}; + simulated blanks {np.mean(eff):.3f}; official real 0.216")
