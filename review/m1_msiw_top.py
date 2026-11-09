import sys; sys.argv=[sys.argv[0]]
import numpy as np, pandas as pd
import m1_msiw as S
R = S.per_window(S.SPEC); T = pd.read_pickle("m1_types.pkl").set_index("k"); W = pd.read_pickle("m1_windows.pkl").set_index("k")
R = R.join(T, on="k").join(W[["far_pred", "far_tp", "far_lab", "n_lab", "date", "tod"]], on="k")
TERMS = [S.window_terms(k) for k in R.k]
for lam in (0.6, 0.8):
    lw = np.array([S.logw(t, lam) for t in TERMS]); w = np.exp(lw - lw.max()); R["w"] = w / w.sum(); R["d"] = R.far - R.iou
    top = R.sort_values("w", ascending=False).head(12)
    print(f"lam {lam}: top-12 weight share {top.w.sum():.3f}, top-30 {R.w.nlargest(30).sum():.3f}")
    print(top[["k", "panel", "date", "tod", "type", "w", "iou", "far", "d", "far_pred", "far_tp", "far_lab", "n_lab"]].round(3).to_string(index=False))
    print(R.groupby("type").w.sum().round(3).to_dict())
