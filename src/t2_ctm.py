"""CTM rollout features for Task 2 (location-agnostic physics forecast).

From the last history row (T-5, the T row is not used) the corridor is simulated 35 min ahead with the cell
transmission model on the released per-link triangular FD. Links shorter than LMIN are merged into macro cells
(capacity = member minimum, free speed = travel-time mean, jam density = length mean) so the CFL step is ~5 s.
Upstream inflow = mean observed flow of the first links over the last 15 min; ramps are ignored (their flows
are small and uncorrelated with mainline changes in this release). Output per link: simulated speed ratio at
T+10/20/30, density ratio k/k_c at T+30 and the first horizon (5..30 min, 35 = never) with v <= v_cut.
"""
import numpy as np

LMIN = 0.2
HOR = (5, 10, 15, 20, 25, 30)
NAMES = ["ctm_vr10", "ctm_vr20", "ctm_vr30", "ctm_kr30", "ctm_first"]
_CELLS = {}


def cells(P):
    key = (P.panel, P.n)
    if key in _CELLS:
        return _CELLS[key]
    L = np.maximum(P.length, 1e-3); g = np.zeros(P.n, int); c = 0; acc = 0.0
    for i in range(P.n):
        g[i] = c; acc += L[i]
        if acc >= LMIN and i < P.n - 1:
            c += 1; acc = 0.0
    if acc < LMIN and c > 0:                       # short tail -> merge into the previous cell
        g[g == c] = c - 1
    nc = g.max() + 1
    Lc = np.bincount(g, L, nc)
    vf = Lc / np.bincount(g, L / P.vf, nc)
    cap = np.full(nc, np.inf); np.minimum.at(cap, g, P.cap)
    kj = np.bincount(g, P.kj * L, nc) / Lc
    kc = cap / vf; kj = np.maximum(kj, kc * 1.05); w = cap / (kj - kc)
    dt = 0.9 * float(np.min(Lc / vf))              # hours
    _CELLS[key] = (g, nc, Lc, vf, cap, kj, kc, w, dt)
    return _CELLS[key]


def ctm_features(P, V, Q, tl):
    """V, Q: day arrays up to row tl (row tl = T, may be NaN); uses rows <= tl-1."""
    g, nc, Lc, vf, cap, kj, kc, w, dt = cells(P)
    last = tl - 1
    k = np.full(P.n, np.nan)
    for r in (last, last - 1, last - 2):           # newest finite observation per link
        if r < 0:
            break
        kk = Q[r] / np.maximum(V[r], 1.0)
        k = np.where(np.isfinite(k), k, kk)
    ok = np.isfinite(k)
    if ok.sum() >= 2:
        xi = np.arange(P.n); k = np.where(ok, k, np.interp(xi, xi[ok], k[ok]))
    else:
        k = P.kc * 0.5
    kc_cell = np.bincount(g, k * P.length.clip(1e-3), nc) / Lc
    kc_cell = np.clip(kc_cell, 0, kj)
    qin = np.nanmean(Q[max(last - 2, 0):last + 1, :3]) if np.isfinite(Q[max(last - 2, 0):last + 1, :3]).any() else cap[0] * 0.5
    qin = float(np.clip(qin, 0, cap[0]))
    rec = {}
    steps = int(np.ceil((35 / 60) / dt)); kcur = kc_cell.copy()
    marks = {int(round(((h + 5) / 60) / dt)): h for h in HOR}
    for s in range(1, steps + 1):
        S = np.minimum(vf * kcur, cap); R = np.minimum(cap, w * (kj - kcur))
        f = np.minimum(S[:-1], R[1:])
        fin = min(qin, R[0]); fout = S[-1]
        dk = np.empty(nc)
        dk[0] = fin - f[0]; dk[1:-1] = f[:-1] - f[1:]; dk[-1] = f[-1] - fout
        kcur = np.clip(kcur + dt * dk / Lc, 0, kj)
        if s in marks:
            rec[marks[s]] = kcur.copy()
    for h in HOR:                                   # rounding safety
        rec.setdefault(h, kcur.copy())
    out = []
    vcut_r = P.vcut / P.vf
    first = np.full(P.n, 35.0)
    vr = {}
    for h in HOR:
        kk = rec[h]
        v = np.where(kk <= kc, vf, w * (kj - kk) / np.maximum(kk, 1e-6))
        vr[h] = (v / vf)[g]
        first = np.where((first == 35.0) & (vr[h] <= vcut_r), h, first)
    out = [vr[10], vr[20], vr[30], (rec[30] / kc)[g], first]
    return np.column_stack(out).astype(np.float32), NAMES
