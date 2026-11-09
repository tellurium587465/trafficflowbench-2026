"""Review: size of the per-link speed / flow-ratio redraw in validation vs train, against the demand features' scale."""
import os, sys, numpy as np
os.environ["TFB_T1_ADAPT"] = "1"
sys.path.insert(0, r"C:/Users/user/tfb/src")
from t1_features2 import link_statics
for p in sys.argv[1:]:
    vt, qt = link_statics(p, "train"); vv, qv = link_statics(p, "validation"); vp, qp = link_statics(p, "private")
    dv, dq = (vv / vt - 1) * 100, (qv / qt - 1) * 100; dvp, dqp = (vp / vt - 1) * 100, (qp / qt - 1) * 100
    print(f"{p}: speed level val-train sd {dv.std():.2f}% (max {np.abs(dv).max():.2f}%), private sd {dvp.std():.2f}% | "
          f"flow ratio val-train sd {dq.std():.2f}% (max {np.abs(dq).max():.2f}%), private sd {dqp.std():.2f}% | corr(val,private) speed {np.corrcoef(dv,dvp)[0,1]:.2f} flow {np.corrcoef(dq,dqp)[0,1]:.2f}")
