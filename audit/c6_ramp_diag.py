"""C6: does the free-flow flow error of the best Task 1 line up with the released ramp flows (unused by Task 1)?
Regress the flow error (per lane) of free-flow targets on the valid on/off-ramp flows attached at the link and its
neighbours (same time); report R^2 and the RMSE that a perfect use of that signal would remove."""
import sys, glob, os, numpy as np
sys.path.insert(0, r"C:/Users/user/tfb/src"); sys.path.insert(0, r"C:/Users/user/tfb/audit")
from cachelib import load
from c2_decomp import kinds

for f in sorted(glob.glob("C:/Users/user/tfb/audit/sim/*.npz")):
    p = os.path.basename(f)[:-4]; Z = np.load(f); K = kinds(Z)
    P0 = load(p, "train"); feb = np.flatnonzero([s.startswith("2031-02") for s in P0.times]); a, b = feb[0], feb[-1] + 1
    li = {l: i for i, l in enumerate(P0.links)}; n = P0.n
    on = np.zeros((b - a, n)); off = np.zeros((b - a, n)); onv = np.zeros((b - a, n), bool); offv = onv.copy(); has_on = np.zeros(n, bool); has_off = has_on.copy()
    for j, (rt, rl) in enumerate(zip(P0.ramp_type, P0.ramp_link)):
        c = li.get(str(rl))
        if c is None:
            continue
        ok = P0.RE[a:b, j]; fl = np.where(ok, np.nan_to_num(P0.R[a:b, j]), 0.0)
        if str(rt).upper() in ("OR", "ON"):
            on[:, c] += fl; onv[:, c] |= ok; has_on[c] = True
        else:
            off[:, c] += fl; offv[:, c] |= ok; has_off[c] = True
    load.cache_clear()
    m = Z["target"] & Z["E"] & np.isfinite(Z["TV"]) & (K == 4)
    lanes = Z["lanes"][None, :]
    e = ((Z["Q"] - Z["TQ"]) / lanes)
    feats = []
    for d in (-2, -1, 0, 1, 2):
        feats += [np.roll(on, d, axis=1) / lanes, np.roll(off, d, axis=1) / lanes]
    valid = (onv | ~has_on[None, :]) & (offv | ~has_off[None, :])
    sel = m & valid
    X = np.column_stack([F[sel] for F in feats] + [np.ones(sel.sum())]); y = e[sel]
    coef, *_ = np.linalg.lstsq(X, y, rcond=None); r = y - X @ coef
    near_ramp = (has_on | has_off | np.roll(has_on, 1) | np.roll(has_off, 1) | np.roll(has_on, -1) | np.roll(has_off, -1))[None, :] & sel
    print(f"{p:11s} free-flow targets {sel.sum():7d}  flow RMSE/lane {np.sqrt(np.mean(y**2)):5.1f} -> {np.sqrt(np.mean(r**2)):5.1f} with ramp terms "
          f"(R2 {1 - (r**2).sum()/(y**2).sum():.3f});  links with a ramp at/next to them: RMSE {np.sqrt(np.mean(e[near_ramp]**2)):5.1f}, others {np.sqrt(np.mean(e[sel & ~near_ramp]**2)):5.1f}", flush=True)
