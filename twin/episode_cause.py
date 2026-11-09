"""What starts a real congestion episode: the flow, or the clock? (10/14) If the start hazard depends on flow beyond the
time of day, the state before T carries the site of a new month's onset (a mapping that transfers); if it is the clock
(scheduled episodes), only the month's own history can tell. Raw C: data, all T2 panels, all train months.
Units = the generator's blocks (block_structure.py, estimated on all months: a structural map, no labels). Episodes =
8-connected congested components (>= 6 cells); start row = first row; site = block of the most downstream link of the
first row. At-risk rows: 05:00-13:00, the site's block not congested in row t-1 and t (no running episode there).
Logit per panel: start ~ site FE + 15-min slot FE + weekend + z, z = 30-min flow (max over the block's links, row t-1)
standardised within (site, slot). Reports the z coefficient (SE), the likelihood-ratio gain of z, and the share of starts
in the top z quintile (1/5 = no flow effect). Writes work/twin_episode_cause.csv.
usage: python episode_cause.py [panels=ALL T2 panels]"""
import sys, numpy as np, pandas as pd, warnings
from scipy import ndimage
warnings.simplefilter("ignore")
sys.path.insert(0, r"C:/Users/user/tfb/src"); sys.path.insert(0, r"C:/Users/user/tfb/twin")
from cachelib import load, T2_PANELS
from onset_shape3_lib import raw_matrices


def blocks(V, P):
    ok = np.isfinite(V); cong = (V <= P.vcut[None, :]) & ok
    a, b = cong[:, :-1], cong[:, 1:]; both = ok[:, :-1] & ok[:, 1:]
    n_any = ((a | b) & both).sum(0); split = ((a ^ b) & both).sum(0) / np.maximum(n_any, 1)
    tight = (n_any >= 30) & (split <= 0.05); bid = np.zeros(P.n, int)
    for i in range(1, P.n):
        bid[i] = bid[i - 1] if tight[i - 1] else bid[i - 1] + 1
    return bid


def logit(A, y, it=30, ridge=1e-4):
    w = np.zeros(A.shape[1]); w[0] = np.log(y.mean() / (1 - y.mean()))
    for _ in range(it):
        p = 1 / (1 + np.exp(-np.clip(A @ w, -30, 30))); W = p * (1 - p) + 1e-12
        H = A.T @ (A * W[:, None]) + ridge * np.eye(A.shape[1]); w = w + np.linalg.solve(H, A.T @ (y - p) - ridge * w)
    p = 1 / (1 + np.exp(-np.clip(A @ w, -30, 30)))
    ll = float(np.sum(y * np.log(p + 1e-12) + (1 - y) * np.log(1 - p + 1e-12)))
    return w, np.sqrt(np.diag(np.linalg.inv(H))), ll


out = []
for p in (sys.argv[1].split(",") if len(sys.argv) > 1 else T2_PANELS):
    P = load(p, "train"); V, Q, E = raw_matrices(P); bid = blocks(V, P); nb = bid.max() + 1
    cong = (V <= P.vcut[None, :]) & np.isfinite(V)
    lab, n = ndimage.label(cong, structure=np.ones((3, 3))); sl = ndimage.find_objects(lab)
    starts = {}
    for i, s_ in enumerate(sl, 1):
        m = lab[s_] == i
        if m.sum() < 6:
            continue
        r0 = s_[0].start; head = s_[1].start + int(np.flatnonzero(m[0]).max())
        starts.setdefault(int(bid[head]), set()).add(r0)
    bc = np.zeros((P.T, nb), bool)                                          # block congested in row t
    for k in range(nb):
        bc[:, k] = cong[:, bid == k].any(axis=1)
    Qf = np.nan_to_num(Q); Qn = np.isfinite(Q)
    c = np.vstack([np.zeros((1, P.n)), np.cumsum(Qf, axis=0)]); kq = np.vstack([np.zeros((1, P.n)), np.cumsum(Qn, axis=0)])
    ma = (c[6:] - c[:-6]) / np.maximum(kq[6:] - kq[:-6], 1); ma[(kq[6:] - kq[:-6]) < 4] = np.nan
    ma = np.vstack([np.full((5, P.n), np.nan), ma])
    days = P.T // 288; we = np.array([pd.Timestamp(str(x)).dayofweek >= 5 for x in P.day[:days]])
    rows = []
    for k, st in starts.items():
        if len(st) < 10:
            continue
        cols = np.flatnonzero(bid == k); f = np.nanmax(ma[:, cols], axis=1)
        for d in range(days):
            for s in range(60, 156):
                t = d * 288 + s
                if bc[t - 1, k] or (bc[t, k] and t not in st) or not np.isfinite(f[t - 1]):
                    continue
                rows.append((k, s // 3, int(we[d]), f[t - 1], t in st))
    if not rows:
        continue
    R = pd.DataFrame(rows, columns=["site", "slot", "we", "f", "y"])
    R["z"] = R.groupby(["site", "slot"]).f.transform(lambda x: (x - x.mean()) / (x.std() + 1e-6))
    S = pd.get_dummies(R.site, prefix="s", drop_first=True).to_numpy(float); T_ = pd.get_dummies(R.slot, prefix="t", drop_first=True).to_numpy(float)
    A0 = np.column_stack([np.ones(len(R)), S, T_, R.we.to_numpy(float)]); A1 = np.column_stack([A0, R.z.to_numpy(float)])
    y = R.y.to_numpy(float); _, _, ll0 = logit(A0, y); w1, se1, ll1 = logit(A1, y)
    q5 = R.z >= R.z.quantile(0.8)
    out.append(dict(panel=p, sites=R.site.nunique(), at_risk=len(R), starts=int(y.sum()), z_coef=round(float(w1[-1]), 3),
                    z_se=round(float(se1[-1]), 3), LR=round(2 * (ll1 - ll0), 1), top_quintile_share=round(float(y[q5].sum() / y.sum()), 3)))
    print(out[-1], flush=True)
    load.cache_clear(); del V, Q, E, lab, cong, bc, ma
pd.DataFrame(out).to_csv(r"C:/Users/user/tfb/work/twin_episode_cause.csv", index=False)
