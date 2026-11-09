"""Local instruments.

state_score   exact replica of score_task1 on the official template targets (train only).
physics_score S_FD exactly as score_task3 computes it, and a proxy S_LWR whose
              right-hand side is the observed accumulation change dN_obs. The
              organizer boundary fluxes are "projected onto the released
              observations", so a submission that reproduces the observations
              scores ~1 on the proxy (0.96 on the real one).
"""
from __future__ import annotations
import numpy as np
from cachelib import load, FAMILY

REGS = (0, 1, 2)
EMPTY_FLOOR_VPH, EMPTY_SHARE = 50.0, 0.20


def _rowreg(P):
    return np.repeat(P.regime[::288], 288)[: P.T] if len(P.regime) == P.T else P.regime


def state_score(P, Vp, Qp, detail=False):
    reg = P.regime
    out = []
    for r in REGS:
        m = P.target & (reg[:, None] == r) & P.E & np.isfinite(P.V) & np.isfinite(P.Q)
        if not m.any():
            out.append(dict(regime=r, n=0, rmse_v=np.nan, rmse_q=np.nan, S=0.0)); continue
        lanes = np.broadcast_to(P.lanes[None, :], P.V.shape)[m]
        dv = np.nan_to_num(Vp[m], nan=0.0) - P.V[m]
        dq = (np.nan_to_num(Qp[m], nan=0.0) - P.Q[m]) / lanes
        rv, rq = np.sqrt(np.mean(dv * dv)), np.sqrt(np.mean(dq * dq))
        s = 0.54 * max(0, 1 - rv / 25) + 0.46 * max(0, 1 - rq / 600)
        out.append(dict(regime=r, n=int(m.sum()), rmse_v=rv, rmse_q=rq, S=s))
    S = float(np.mean([o["S"] for o in out]))
    return (S, out) if detail else S


def physics_score(P, Vp, Qp, detail=False, validity=None):
    """Return (S_FD, S_LWR_proxy, S_physics_proxy) averaged over regimes."""
    on_v, off_v, _, _ = validity if validity is not None else P.ramp_validity()
    reg = P.regime
    res = []
    for r in REGS:
        day = (reg == r)[:, None]
        field = P.E & day & np.isfinite(P.V) & np.isfinite(P.Q)
        tgt = P.target & field
        V = np.where(tgt, Vp, P.V); Q = np.where(tgt, Qp, P.Q)
        V = np.where(field, V, np.nan); Q = np.where(field, Q, np.nan)
        ok = field & np.isfinite(V) & np.isfinite(Q)
        if not ok.any():
            res.append((0.0, 0.0)); continue
        k = Q / np.maximum(V, 1.0)
        lanes = P.lanes[None, :]
        ql, kl = Q / lanes, k / lanes
        capl, kcl, kjl = (P.cap / P.lanes)[None, :], (P.kc / P.lanes)[None, :], (P.kj / P.lanes)[None, :]
        qfd = np.where(kl <= kcl, P.vf[None, :] * kl, capl / np.maximum(kjl - kcl, 1e-9) * np.maximum(kjl - kl, 0))
        sfd = max(0.0, 1 - np.abs(ql - qfd)[ok].sum() / (np.abs(ql)[ok].sum() + 1e-9))
        if tgt.any() and (np.nan_to_num(Qp[tgt], nan=0) < EMPTY_FLOOR_VPH).mean() > EMPTY_SHARE:
            sfd = 0.0
        N = k * P.length[None, :]
        kt = P.Q / np.maximum(P.V, 1.0)
        Nt = np.where(field, kt * P.length[None, :], np.nan)
        dN = N[1:] - N[:-1]; dNt = Nt[1:] - Nt[:-1]
        tr = ok[1:] & ok[:-1] & on_v[:-1] & off_v[:-1]
        den = np.abs(dNt[tr]).sum()
        e = np.abs(dN[tr] - dNt[tr]).sum() / (den + 1e-9)
        slwr = max(0.0, 1 - min(1.0, e))
        res.append((sfd, slwr))
    sfd = float(np.mean([a for a, _ in res])); slwr = float(np.mean([b for _, b in res]))
    out = (sfd, slwr, (sfd + 2 * slwr) / 3)
    return (out, res) if detail else out


def family_mean(panel_scores: dict) -> float:
    fam = {}
    for p, s in panel_scores.items():
        fam.setdefault(FAMILY[p], []).append(s)
    return float(np.mean([np.mean(v) for v in fam.values()]))
