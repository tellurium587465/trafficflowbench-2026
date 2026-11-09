"""A22 (10/24, audit day 3 remainder): the report's instrument numbers (Section 3, Fig. 3) recomputed from the stored tables
with plain numpy, independent of judge.py / make_report_figs.py:
  calibration: work/t2/g85_calibv3.csv (LB difference vs the instrument's main-weight prediction per submission pair);
               pairs with |dLB| < 0.0005 excluded from the sign count (they only calibrate the size), slope of LB on the
               instrument through the origin, correlation, residual sd, over the included pairs
  A/A:         work/t2/m3v2_aa_q15f_q19f_q24f_q32f_q43f_q46fv3.csv (two-seed arms of the same configuration, instrument v3):
               number of comparisons, share |z| > 2, sd of z, empirical sd of d vs the mean bootstrap SE -> seed-noise sd
usage: python a22_calib_aa_recompute.py"""
import numpy as np, pandas as pd

W = "C:/Users/user/tfb/work/t2/"
C = pd.read_csv(W + "g85_calibv3.csv"); inc = C.lb.abs() >= 0.0005
x, y = C.main[inc].to_numpy(), C.lb[inc].to_numpy()
b = (x * y).sum() / (x * x).sum(); res = y - b * x
print(f"calibration: {len(C)} pairs, {inc.sum()} with |dLB| >= 0.0005 (excluded: {', '.join(C.pair[~inc])})")
print(f"  sign agrees {int((np.sign(x) == np.sign(y)).sum())} of {inc.sum()} | slope (LB on instrument, origin) {b:.2f} | correlation {np.corrcoef(x, y)[0, 1]:.2f} | "
      f"residual sd {res.std(ddof=1):.4f} | negative control: LB {C.lb[C.pair.str.contains('negative')].iloc[0]:+.4f}, instrument {C.main[C.pair.str.contains('negative')].iloc[0]:+.4f}")
print("  reverse slope (instrument on LB, origin) %.2f" % ((x * y).sum() / (y * y).sum()))
A = pd.read_csv(W + "m3v2_aa_q15f_q19f_q24f_q32f_q43f_q46fv3.csv")
sd_d, mse = A.d.std(), np.sqrt((A.se ** 2).mean())
print(f"A/A (v3): {len(A)} comparisons over {A.cfg.nunique()} configurations | share |z| > 2 {np.mean(np.abs(A.z) > 2):.2f} | sd of z {A.z.std():.2f} | "
      f"sd of d {sd_d:.4f}, mean SE {mse:.4f} -> seed-noise sd {np.sqrt(max(sd_d ** 2 - mse ** 2, 0)):.4f}")
