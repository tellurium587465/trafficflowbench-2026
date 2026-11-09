"""Digital twin stage 1 (POLICY_20261012 §3): can a cell transmission model (CTM) with the released fundamental
diagram replay where and when queues form on TRAIN days? Inputs per day: the observed upstream boundary flow (link 0),
the observed on/off-ramp counts, fd_parameters (capacity, free speed, jam density). No incidents, no fitted demand.
Short links are merged into cells of >= LMIN km (cell capacity = min of its links, travel time additive), dt = 2 s,
all days simulated at once (arrays days x cells). Output: per day the 5-min speed of every link (cell value), the
queue indicator v <= v_cut, compared with the observed queue indicator: cell IoU over the day, recall/precision of
queued link-slots, and the agreement of the first queue head of the day (within +-3 links, +-30 min).
Only train data is used (validation/private months are never touched by the twin).

usage: python ctm_replay.py <panel> [n_days=20] [LMIN=0.12]
"""
import sys, time, numpy as np, pandas as pd
sys.path.insert(0, r"C:/Users/user/tfb/src")
from cachelib import load

panel = sys.argv[1]; NDAYS = int(sys.argv[2]) if len(sys.argv) > 2 else 20; LMIN = float(sys.argv[3]) if len(sys.argv) > 3 else 0.12
DT = 2.0 / 3600.0                                              # hours
P = load(panel, "train"); n = P.n; t0 = time.time()

# ---------------- cells ----------------
L = np.nan_to_num(P.length, nan=0.1); cell_of = np.zeros(n, int); c = 0; acc = 0.0
for i in range(n):
    cell_of[i] = c; acc += L[i]
    if acc >= LMIN and i < n - 1:
        c += 1; acc = 0.0
NC = cell_of.max() + 1
cl = np.bincount(cell_of, weights=L, minlength=NC)
tt = np.bincount(cell_of, weights=L / P.vf, minlength=NC)       # travel time (h) at free speed
vf = cl / tt
cap = np.array([P.cap[cell_of == j].min() for j in range(NC)])
kj = np.array([np.average(P.kj[cell_of == j], weights=L[cell_of == j]) for j in range(NC)])
kc = cap / vf; w = cap / np.maximum(kj - kc, 1e-6)
DT = min(DT, 0.9 * (cl / np.maximum(vf, w)).min())                  # CFL
sub = int(np.ceil(300 / (DT * 3600))); DT = 300 / sub / 3600.0

# ---------------- junction ratios (v2): r_i(daytype, tod) = q_{i+1} / q_i where both links are free-flowing and observed,
# median over train days, smoothed over +-15 min. Captures the observed ramps AND the unobserved connectors (CONN), whose
# net flows can be thousands of vph (D12_I5_S link 90 -> 91: -3400). Applied at cell exits as the product of the ratios
# of the junctions inside / at the end of the cell. ----------------
ffree = np.isfinite(P.V) & np.isfinite(P.Q) & (P.V > 0.85 * P.vf[None, :]) & (P.Q > 100)
dayt = np.array([pd.Timestamp(str(x)).dayofweek >= 5 for x in P.day]).repeat(288)[:P.T].astype(int)
tod = np.arange(P.T) % 288
ok = ffree[:, :-1] & ffree[:, 1:]
lr = np.where(ok, np.log(np.maximum(P.Q[:, 1:], 1) / np.maximum(P.Q[:, :-1], 1)), np.nan)        # (T, n-1)
RAT = np.zeros((2, 288, n - 1))
for dt_ in (0, 1):
    m = dayt == dt_
    g = pd.DataFrame(lr[m]).groupby(tod[m]).median().reindex(range(288))
    g = g.rolling(7, center=True, min_periods=1).median().ffill().bfill().fillna(0.0)
    RAT[dt_] = np.exp(g.to_numpy())
# cell-exit ratio: product over junctions i whose upstream link belongs to cell j (junction i = between link i and i+1)
jcell = cell_of[:-1]
CR = np.ones((2, 288, NC))
for j in range(NC):
    sel = jcell == j
    if sel.any():
        CR[:, :, j] = np.prod(RAT[:, :, sel], axis=2)
CR[:, :, -1] = 1.0
# ---------------- days: weekdays with an observed queue somewhere ----------------
Qm = (np.nan_to_num(P.V, nan=1e9) <= P.vcut[None, :]) & P.E
D = P.T // 288; days = [d for d in range(D) if pd.Timestamp(str(P.day[d])).dayofweek < 5 and Qm[d * 288:(d + 1) * 288].any()]
days = days[:: max(1, len(days) // NDAYS)][:NDAYS]
up = P.Q[:, 0].astype(float)
up = pd.Series(up).interpolate(limit_direction="both").to_numpy()

# ---------------- simulate all days at once ----------------
nd = len(days); k = np.zeros((nd, NC)); qin = np.zeros(nd)            # density veh/km, upstream virtual queue (veh)
Vs = np.zeros((nd, 288, NC)); Fs = np.zeros((nd, 288, NC))
rows0 = np.array([d * 288 for d in days])
for r in range(288):
    dem = up[rows0 + r]; dty = dayt[rows0 + r]; cr = CR[dty, r]                    # (nd, NC)
    fl_acc = np.zeros((nd, NC)); k_acc = np.zeros((nd, NC))
    for _ in range(sub):
        S = np.minimum(vf * k, cap)                                   # sending
        R = np.minimum(cap, w * np.maximum(kj - k, 0))                # receiving
        # cell exit ratio cr_j: flow leaving cell j is S_j; flow arriving in j+1 is cr_j * that (cr<1: diverge, the
        # rest exits; cr>1: merge, the extra enters). The upstream move is limited so that the arrival fits R_{j+1}.
        crj = cr[:, :-1]
        f = np.minimum(S[:, :-1], R[:, 1:] / np.maximum(crj, 1e-3))  # leaving j toward j+1
        arr = crj * f                                                 # arriving in j+1
        inflow0 = np.minimum(dem + qin / DT, R[:, 0])
        qin = np.maximum(qin + (dem - inflow0) * DT, 0)
        out_last = S[:, -1]
        inn = np.concatenate([inflow0[:, None], arr], axis=1)
        out = np.concatenate([f, out_last[:, None]], axis=1)
        k = np.maximum(k + DT / cl * (inn - out), 0)
        fl_acc += out; k_acc += k
    fl = fl_acc / sub; kk = k_acc / sub
    Fs[:, r] = fl; Vs[:, r] = np.where(kk > 1e-3, np.minimum(fl / np.maximum(kk, 1e-6), vf), vf)

# ---------------- compare on links ----------------
res = []
for a, d in enumerate(days):
    vs = Vs[a][:, cell_of]; qs = vs <= P.vcut[None, :]
    qo = Qm[d * 288:(d + 1) * 288]; el = P.E[d * 288:(d + 1) * 288]
    qs = qs & el
    u = (qs | qo).sum(); i_ = (qs & qo).sum()
    def first_head(Q):
        rr = np.flatnonzero(Q.any(axis=1))
        if not len(rr):
            return None
        r0 = rr[0]; return r0, np.flatnonzero(Q[r0]).max()
    ho, hs = first_head(qo), first_head(qs)
    hit = ho is not None and hs is not None and abs(ho[0] - hs[0]) <= 6 and abs(ho[1] - hs[1]) <= 3
    res.append(dict(day=P.day[d], obs=int(qo.sum()), sim=int(qs.sum()), iou=i_ / u if u else 1.0, recall=i_ / max(qo.sum(), 1),
                    precision=i_ / max(qs.sum(), 1), first_head_hit=hit, ho=ho, hs=hs))
R_ = pd.DataFrame(res)
print(f"{panel}: {NC} cells from {n} links, {nd} weekdays, {time.time()-t0:.0f}s")
print(R_[["iou", "recall", "precision", "first_head_hit"]].mean().round(3).to_dict(), "| queued link-slots obs/sim per day",
      int(R_.obs.mean()), int(R_.sim.mean()))
print(R_[["day", "obs", "sim", "iou", "ho", "hs"]].head(8).to_string(index=False))
