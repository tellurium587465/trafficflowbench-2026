"""Cell transmission model core for the digital twin (POLICY_20261012 §3). Validated in stage 1 (ctm_replay.py):
with the released fundamental diagram and junction ratios measured on train, replaying train weekdays gives queue
IoU 0.40 and the right volume (D12_I5_S, no incidents).

Model: links merged into cells of >= LMIN km (capacity = min of the links, travel time additive); Godunov/CTM fluxes
with a triangular FD; each cell exit applies a ratio cr (q_next / q_this; < 1 diverge, > 1 merge); upstream boundary
with a virtual queue; free outflow downstream. All days run at once (arrays days x cells).
"""
import numpy as np, pandas as pd


class Corridor:
    def __init__(self, P, lmin=0.12, cap_mult=None, cell_of=None):
        """cell_of: optional (n,) non-decreasing cell index per link (10/14: the generator's blocks, block_structure.py);
        default = links merged into cells of >= lmin km."""
        n = P.n; L = np.nan_to_num(P.length, nan=0.1)
        if cell_of is None:
            cell_of = np.zeros(n, int); c = 0; acc = 0.0
            for i in range(n):
                cell_of[i] = c; acc += L[i]
                if acc >= lmin and i < n - 1:
                    c += 1; acc = 0.0
        cell_of = np.asarray(cell_of, int)
        self.cell_of = cell_of; self.NC = NC = cell_of.max() + 1
        cap_link = P.cap * (cap_mult if cap_mult is not None else 1.0)
        self.cl = np.bincount(cell_of, weights=L, minlength=NC)
        tt = np.bincount(cell_of, weights=L / P.vf, minlength=NC)
        self.vf = self.cl / tt
        self.cap = np.array([cap_link[cell_of == j].min() for j in range(NC)])
        self.kj = np.array([np.average(P.kj[cell_of == j], weights=L[cell_of == j]) for j in range(NC)])
        self.w = self.cap / np.maximum(self.kj - self.cap / self.vf, 1e-6)
        dt = min(2.0 / 3600, 0.9 * (self.cl / np.maximum(self.vf, self.w)).min())
        self.sub = int(np.ceil(300 / (dt * 3600))); self.dt = 300 / self.sub / 3600.0

    def cell_ratios(self, RAT):
        """junction ratios (..., n-1) -> cell-exit ratios (..., NC): product over the junctions owned by each cell."""
        jcell = self.cell_of[:-1]; CR = np.ones(RAT.shape[:-1] + (self.NC,))
        for j in range(self.NC):
            sel = jcell == j
            if sel.any():
                CR[..., j] = np.prod(RAT[..., sel], axis=-1)
        CR[..., -1] = 1.0
        return CR

    def run(self, up, CR, capf=None, snapshot=False, cdrop=1.0, hook=None):
        """up: (nd, 288) upstream demand (vph); CR: (nd, 288, NC) cell-exit ratios; capf: optional (nd, 288, NC)
        capacity factors (incidents). Returns 5-min mean speed and flow per cell, (nd, 288, NC) each.
        10/14 options (defaults = the original model): snapshot=True records the state at the END of each 5-min row
        instead of the mean; cdrop < 1 = capacity drop: a cell whose downstream neighbour is congested (density above
        critical) discharges at cdrop x capacity (two-capacity bottleneck). hook(r, V, F, k) -> (nd, NC) extra capacity
        factors for row r, computed from the rows already simulated (breakdown models; onset_shape3.py)."""
        nd = up.shape[0]; NC = self.NC; dt = self.dt
        k = np.zeros((nd, NC)); qin = np.zeros(nd)
        V = np.zeros((nd, 288, NC)); F = np.zeros((nd, 288, NC))
        for r in range(288):
            dem = up[:, r]; cr = CR[:, r]; crj = cr[:, :-1]
            cap = self.cap[None, :] * (capf[:, r] if capf is not None else 1.0)
            if hook is not None:
                cap = cap * hook(r, V, F, k)
            fl_acc = np.zeros((nd, NC)); k_acc = np.zeros((nd, NC))
            for _ in range(self.sub):
                if cdrop < 1.0:                                   # congested cell -> reduced discharge
                    cong = k > (self.cap / self.vf)[None, :]
                    capx = np.where(cong, cap * cdrop, cap)
                else:
                    capx = cap
                S = np.minimum(self.vf * k, capx)
                R = np.minimum(cap, self.w * np.maximum(self.kj - k, 0))
                f = np.minimum(S[:, :-1], R[:, 1:] / np.maximum(crj, 1e-3))
                inflow0 = np.minimum(dem + qin / dt, R[:, 0])
                qin = np.maximum(qin + (dem - inflow0) * dt, 0)
                inn = np.concatenate([inflow0[:, None], crj * f], axis=1)
                out = np.concatenate([f, S[:, -1:]], axis=1)
                k = np.maximum(k + dt / self.cl * (inn - out), 0)
                fl_acc += out; k_acc += k
            fl, kk = (out, k) if snapshot else (fl_acc / self.sub, k_acc / self.sub)
            F[:, r] = fl; V[:, r] = np.where(kk > 1e-3, np.minimum(fl / np.maximum(kk, 1e-6), self.vf), self.vf)
        return V, F


def junction_ratios(P, rows=None):
    """median log ratio q_{i+1}/q_i on free-flowing observed pairs, by (weekend, 5-min slot), +-15 min smoothing.
    rows: optional boolean mask of train rows to estimate from. Returns RAT (2, 288, n-1)."""
    ff = np.isfinite(P.V) & np.isfinite(P.Q) & (P.V > 0.85 * P.vf[None, :]) & (P.Q > 100)
    if rows is not None:
        ff = ff & rows[:, None]
    dayt = np.array([pd.Timestamp(str(x)).dayofweek >= 5 for x in P.day]).repeat(288)[:P.T].astype(int)
    tod = np.arange(P.T) % 288
    ok = ff[:, :-1] & ff[:, 1:]
    lr = np.where(ok, np.log(np.maximum(P.Q[:, 1:], 1) / np.maximum(P.Q[:, :-1], 1)), np.nan)
    RAT = np.zeros((2, 288, P.n - 1))
    for d in (0, 1):
        m = dayt == d
        g = pd.DataFrame(lr[m]).groupby(tod[m]).median().reindex(range(288))
        g = g.rolling(7, center=True, min_periods=1).median().ffill().bfill().fillna(0.0)
        RAT[d] = np.exp(g.to_numpy())
    return RAT
