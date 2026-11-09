"""D0: test-time selector for 'boundary' Task 1 targets = released observations within +-R links / +-S steps contain
both congested (v < 0.6 vf) and free (v >= 0.6 vf) cells; targets inside the T2 blanks are excluded.
Coverage of the error of the current best Task 1 (simulated months, audit/sim)."""
import sys, glob, os, numpy as np
sys.path.insert(0, r"C:/Users/user/tfb/audit")
from c2_decomp import kinds


def mixed(Z, R=2, S=2):
    rel = np.where(Z["blank"], np.nan, Z["TV"]); vf = Z["vf"][None, :]
    c = np.where(np.isfinite(rel), rel < 0.6 * vf, False); f = np.where(np.isfinite(rel), rel >= 0.6 * vf, False)
    cc = np.zeros_like(c); ff = np.zeros_like(f)
    for dj in range(-R, R + 1):
        for dt in range(-S, S + 1):
            if dj == 0 and dt == 0:
                continue
            cc |= np.roll(np.roll(c, dt, 0), dj, 1); ff |= np.roll(np.roll(f, dt, 0), dj, 1)
    return cc & ff


for R, S in ((1, 1), (2, 2), (3, 2)):
    tot = dict(n=0, sel=0, sev=0, sev_sel=0, seq=0, seq_sel=0, bnd=0, bnd_sel=0)
    for f in sorted(glob.glob("C:/Users/user/tfb/audit/sim/*.npz")):
        Z = np.load(f); K = kinds(Z)
        m = Z["target"] & Z["E"] & np.isfinite(Z["TV"]) & ~Z["gap"]
        sel = m & mixed(Z, R, S)
        sev = (Z["V"] - Z["TV"]) ** 2; seq = ((Z["Q"] - Z["TQ"]) / Z["lanes"][None, :]) ** 2
        tot["n"] += m.sum(); tot["sel"] += sel.sum(); tot["sev"] += sev[m].sum(); tot["sev_sel"] += sev[sel].sum()
        tot["seq"] += seq[m].sum(); tot["seq_sel"] += seq[sel].sum(); tot["bnd"] += (m & (K == 2)).sum(); tot["bnd_sel"] += (sel & (K == 2)).sum()
    print(f"R={R} S={S}: selected {tot['sel']/tot['n']:.2%} of non-blank targets, holding {tot['sev_sel']/tot['sev']:.1%} of speed SE and "
          f"{tot['seq_sel']/tot['seq']:.1%} of flow SE; covers {tot['bnd_sel']/tot['bnd']:.1%} of truth-boundary targets")
