"""E[max] of the two selected submissions (11/04, ledgers #214 / #179 (F)): Kaggle scores the better of the two final picks, so a pair
is worth E[max(S_a, S_b)] over the draw of the private windows. Per draw: NO onset + NG ongoing windows with replacement, a candidate's
T2 = 0.5 mean onset IoU + 0.5 mean ongoing IoU, S = 0.30 T2 (Tasks 1, 3, 4 equal across the candidates). The uncertainty of the
pair's gain over the reference pair comes from an outer bootstrap over the windows (n_out) with n_in inner draws each.
cands: {name: (onset column or None for ongoing-only, ongoing column)}; pairs: [(a, b)]; ref: the reference pair."""
import numpy as np


def emax_table(on, og, cands, pairs, ref, NO=40, NG=40, n_out=200, n_in=200, seed=0):
    rng = np.random.default_rng(seed); Xo = {c: (on[v[0]].to_numpy(float) if (on is not None and v[0]) else None) for c, v in cands.items()}; Xg = {c: og[v[1]].to_numpy(float) for c, v in cands.items()}
    no, ng = (len(on) if on is not None else 0), len(og); res = {p: [] for p in pairs}; lev = {p: [] for p in pairs}
    for _ in range(n_out):
        io = rng.integers(0, no, no) if no else None; ig = rng.integers(0, ng, ng)
        do = rng.integers(0, no, (n_in, NO)) if no else None; dg = rng.integers(0, ng, (n_in, NG))
        S = {}
        for c in cands:
            t = Xg[c][ig][dg].mean(axis=1)
            if Xo[c] is not None:
                t = 0.5 * Xo[c][io][do].mean(axis=1) + 0.5 * t
            S[c] = 0.30 * t
        r = np.maximum(S[ref[0]], S[ref[1]]).mean()
        for p in pairs:
            e = np.maximum(S[p[0]], S[p[1]]).mean(); res[p].append(e - r); lev[p].append(e)
    return {p: (float(np.mean(lev[p])), float(np.mean(res[p])), float(np.std(res[p]))) for p in pairs}
