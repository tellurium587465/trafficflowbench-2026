"""Task 4 ODME estimators.

All estimators fit the released counts exactly (A f = c, f >= 0) where they can
and differ only in how "close to the prior" is measured. The dual is solved over
the distinct rows of A (links with the same path set carry one constraint).
"""
from __future__ import annotations
import numpy as np, pandas as pd
from scipy import sparse
from scipy.optimize import minimize
from common import REL, PANELS


def load_problem(panel, split):
    nd = REL / "corridors" / panel / "network"
    paths = pd.read_csv(nd / "path_set.csv", dtype=str)
    inc = pd.read_csv(nd / "path_link_incidence.csv", dtype=str).drop_duplicates(["link_id", "path_id"])
    folder = REL / "task4" / panel / split
    prior = pd.read_csv(folder / "synthetic_weak_prior.csv", dtype={"path_id": str, "departure_time": str})
    counts = pd.read_csv(folder / "synthetic_link_counts.csv", dtype={"link_id": str})
    ids = paths.path_id.tolist(); pi = {p: i for i, p in enumerate(ids)}
    supported = set(inc.link_id)
    links = [l for l in counts.link_id if l in supported]
    li = {l: i for i, l in enumerate(links)}
    m = inc[inc.link_id.isin(li)]
    A = sparse.csr_matrix((np.ones(len(m)), (m.link_id.map(li), m.path_id.map(pi))), shape=(len(links), len(ids)))
    c = counts.set_index("link_id").reindex(links)["count"].to_numpy(float)
    b = prior.set_index("path_id").reindex(ids).path_flow.to_numpy(float)
    return dict(panel=panel, split=split, ids=ids, paths=paths, links=links, A=A, c=c, b=b,
                departure=str(prior.departure_time.iloc[0]))


def distinct_rows(A, c):
    """Collapse identical rows of A (same path set); return reduced A, c and the spread of c inside groups."""
    Ad = A.tocsr(); keys = {}
    groups = []
    for i in range(Ad.shape[0]):
        k = tuple(Ad.indices[Ad.indptr[i]:Ad.indptr[i + 1]].tolist())
        if k not in keys:
            keys[k] = len(groups); groups.append([i])
        else:
            groups[keys[k]].append(i)
    rows = [g[0] for g in groups]
    cr = np.array([c[g].mean() for g in groups])
    spread = max((np.ptp(c[g]) for g in groups), default=0.0)
    return Ad[rows], cr, spread, groups


def solve_weighted_l2(A, c, b, wdiag, tol=1e-10, maxiter=20000):
    """min 0.5 sum w (f-b)^2  s.t. A f = c, f >= 0   via the concave dual in y."""
    winv = 1.0 / wdiag
    def neg_dual(y):
        u = b + winv * (A.T @ y)
        f = np.maximum(u, 0)
        # dual value: -[ 0.5 sum w (f-b)^2 + y (c - A f) ]  (negated for minimization)
        val = 0.5 * np.sum(wdiag * (f - b) ** 2) + y @ (c - A @ f)
        grad = c - A @ f
        return -val, -grad
    y0 = np.zeros(A.shape[0])
    r = minimize(neg_dual, y0, jac=True, method="L-BFGS-B",
                 options=dict(maxiter=maxiter, ftol=1e-16, gtol=tol, maxcor=50))
    f = np.maximum(b + winv * (A.T @ r.x), 0)
    return f, r


def solve_kl(A, c, b, maxiter=20000):
    """min sum f log(f/b) - f + b  s.t. A f = c   ->  f = b * exp(A^T y)."""
    def neg_dual(y):
        z = np.clip(A.T @ y, -50, 50)
        f = b * np.exp(z)
        val = c @ y - f.sum()
        grad = c - A @ f
        return -val, -grad
    r = minimize(neg_dual, np.zeros(A.shape[0]), jac=True, method="L-BFGS-B",
                 options=dict(maxiter=maxiter, ftol=1e-16, gtol=1e-10, maxcor=50))
    return b * np.exp(np.clip(A.T @ r.x, -50, 50)), r


def solve_ridge(A, c, b, lam=0.05, f0=None, maxiter=20000):
    """The baseline objective 0.5||Af-c||^2 + 0.5 lam ||f-b||^2, f >= 0, solved tightly."""
    def obj(f):
        res, ch = A @ f - c, f - b
        return 0.5 * (res @ res + lam * ch @ ch), A.T @ res + lam * ch
    x0 = np.maximum(b, 0) if f0 is None else f0
    r = minimize(obj, x0, jac=True, method="L-BFGS-B", bounds=[(0, None)] * len(b),
                 options=dict(maxiter=maxiter, ftol=1e-16, gtol=1e-8, maxcor=50))
    return np.maximum(r.x, 0), r


def s_link(prob, f):
    return max(0.0, 1 - np.abs(prob["A"] @ f - prob["c"]).sum() / max(prob["c"].sum(), 1e-9))


def frame(prob, f):
    x = prob["paths"][["path_id", "origin_zone", "destination_zone"]].copy()
    x.insert(0, "departure_time", prob["departure"]); x.insert(0, "panel", prob["panel"])
    x["path_flow"] = f
    return x


def solve_l2_exact(A, c, b, y0=None, iters=100, tol=1e-12):
    """Exact Euclidean projection of b onto {Af=c, f>=0} by semismooth Newton on the dual.

    f(y) = max(0, b + A^T y); solve A f(y) = c. Each step solves (A_S A_S^T) dy = c - A f(y)
    on the current support S, with a backtracking line search on ||residual||.
    """
    A = A.tocsr(); At = A.T.tocsr()
    y = np.zeros(A.shape[0]) if y0 is None else y0.copy()
    def resid(y):
        f = np.maximum(b + At @ y, 0)
        return f, c - A @ f
    f, r = resid(y)
    for it in range(iters):
        nr = np.abs(r).max()
        if nr < tol * max(1.0, np.abs(c).max()):
            break
        S = (b + At @ y) > 0
        AS = A[:, S]
        M = (AS @ AS.T).toarray() + 1e-12 * np.eye(A.shape[0])
        dy = np.linalg.solve(M, r)
        step = 1.0
        while step > 1e-8:
            yn = y + step * dy
            fn, rn = resid(yn)
            if np.abs(rn).max() < nr:
                y, f, r = yn, fn, rn; break
            step *= 0.5
        else:
            break
    return f, y, it, np.abs(r).max()
