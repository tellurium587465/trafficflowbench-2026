"""M1: map-shift importance weighting (MSIW) for the ongoing yardstick, train OOF only.

Model of the test months (README: own demand draw -> queue map out of the train range): the activation probability of a
far site moves from its train location-time prior U_s toward a flat rate u_bar:  P'_s = (1-lam) U_s + lam u_bar.
Each selector-like ongoing window gets the likelihood ratio of its far-site outcomes y_s (3-link buckets > 3 links from
the queue at T; y_s = 1 if the label queues there in T+5..T+30):
    log w = sum_s y_s log(P'_s / U_s) + (1 - y_s) log((1 - P'_s) / (1 - U_s))
so windows where a usual far site stays free (memory would be wrong) gain weight and windows where it activates as usual
lose weight. lam = 0 is the plain month CV. Reported per lam: weighted ongoing IoU, IoU after the far rule (D=3),
their difference, ESS. Anchors: LB ongoing 0.822, far rule +0.021.
usage: python m1_msiw.py [spec]
"""
import os, sys
os.environ["TFB_T2CACHE"] = "2_noT_af"; os.environ["TFB_T2_FAR"] = "3"
sys.path.insert(0, r"C:/Users/user/tfb/src")
import numpy as np, pandas as pd
import t2_eval_oof as EV
from t2_model import best_set
from cachelib import load, T2_PANELS
from t2_windows import queue_matrix, iou

SPEC = sys.argv[1] if len(sys.argv) > 1 else "l127_full_af:2+q14_attn1_glob+q16_attnglob_s1+q14_attn1_wp05:2+q13_attn1:2+q12_af_r13_nopos+q17_drop1"
M, LAB = EV.T.META, EV.T.LAB
U = {}
for p in T2_PANELS:
    P = load(p, "train"); D = P.T // 288; qb = queue_matrix(P).reshape(D, 48, 6, P.n).any(axis=2)
    days = [str(x) for x in P.day]; mon = np.array([d[:7] for d in days]); wk = np.array([pd.Timestamp(d).dayofweek >= 5 for d in days])
    for m in sorted(set(mon)):
        for we in (False, True):
            U[(p, m, we)] = qb[(mon != m) & (wk == we)].mean(axis=0)
    load.cache_clear()


def window_terms(k):
    """far regions: runs of far links with U >= 0.05 (gap <= 1) + one 'elsewhere' region (far links with U < 0.05).
    Relocation shift keeps the expected far activation mass rho = sum_r U_r and spreads a share lam of it over the far
    links in proportion to their length (count): P'_r = (1 - lam) U_r + lam * min(rho, 1) * n_r / n_far."""
    lab, el = LAB[k]; far = EV.far_mask(k); n = len(far)
    d = str(M["date"][k]); we = pd.Timestamp(d).dayofweek >= 5; b = min((int(M["t"][k]) % 288 + 3) // 6, 47)
    Uk = U[(M["panel"][k], d[:7], we)][b]
    q = (lab.astype(bool) & el.astype(bool)).any(axis=0)
    hi = far & (Uk >= 0.05); regs = []
    for j in np.flatnonzero(hi):
        if regs and j - regs[-1][-1] <= 2:
            regs[-1].append(j)
        else:
            regs.append([j])
    Ur, yr, nr = [], [], []
    for r in regs:
        sl = slice(r[0], r[-1] + 1); Ur.append(Uk[sl].max()); yr.append(q[sl][far[sl]].any()); nr.append(far[sl].sum())
    lo = far & (Uk < 0.05)
    if lo.any():
        Ur.append(1 - np.prod(1 - Uk[lo])); yr.append(q[lo].any()); nr.append(lo.sum())
    return np.array(Ur), np.array(yr, bool), np.array(nr, float)


def logw(terms, lam):
    Ur, yr, nr = terms
    if not len(Ur):
        return 0.0
    Pu = np.clip(Ur, 0.02, 0.98); rho = min(Pu.sum(), 1.0); Pp = np.clip((1 - lam) * Pu + lam * rho * nr / nr.sum(), 0.02, 0.98)
    return float(np.sum(np.where(yr, np.log(Pp / Pu), np.log((1 - Pp) / (1 - Pu)))))


def per_window(spec):
    oof = EV.load_oof(spec); rows = []
    for k in np.flatnonzero(M["is_sel"] & (M["cond"] == "queue_ongoing")):
        o = oof[k]; s = best_set(o.ravel(), np.ones(o.size, bool)).reshape(o.shape); far = EV.far_mask(k)
        lab, el = LAB[k]; lab = lab.astype(bool); el = el.astype(bool)
        rows.append(dict(k=int(k), panel=M["panel"][k], cl=f"{M['panel'][k]}|{str(M['date'][k])[:10]}",
                         iou=iou(s, lab, el), far=iou(s & ~far[None, :], lab, el)))
    return pd.DataFrame(rows)


def wmean(x, w):
    return float(np.sum(w * x) / np.sum(w))


def boot(R, w, cols, reps=300, seed=0):
    rng = np.random.default_rng(seed); u = R.cl.unique(); idx = {c: np.flatnonzero(R.cl.to_numpy() == c) for c in u}; out = []
    for _ in range(reps):
        ii = np.concatenate([idx[c] for c in rng.choice(u, len(u))]); out.append([wmean(R[c].to_numpy()[ii], w[ii]) for c in cols])
    return np.std(out, axis=0)


if __name__ == "__main__":
    specs = sys.argv[1:] or [SPEC]
    base = per_window(specs[0]); TERMS = [window_terms(k) for k in base.k]
    LAMS = (0.0, 0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0)
    WS = {}
    for lam in LAMS:
        lw = np.array([logw(t, lam) for t in TERMS]); lw -= lw.max(); WS[lam] = np.exp(lw)
    base["d"] = base.far - base.iou
    print(f"ref {specs[0][:70]}  windows {len(base)}")
    print(" lam   ESS   ongoing  far-rule  d(far) +- SE(cluster boot)")
    for lam in LAMS:
        w = WS[lam]; se = boot(base, w, ["d"])[0]
        print(f" {lam:.1f} {w.sum()**2/np.sum(w**2):6.0f}   {wmean(base.iou, w):.4f}   {wmean(base.far, w):.4f}   {wmean(base.d, w):+.4f} +- {se:.4f}", flush=True)
    for sp in specs[1:]:
        R = per_window(sp); R["diff"] = R.iou.to_numpy() - base.iou.to_numpy(); R["dfar"] = R.far - R.iou
        line = f"{sp[:44]:44s}"
        for lam in (0.0, 0.6, 0.8):
            w = WS[lam]; se = boot(R, w, ["diff"])[0]
            line += f" | lam{lam:.1f} {wmean(R.iou, w):.4f} diff {wmean(R['diff'], w):+.4f}({wmean(R['diff'], w)/max(se,1e-9):+.1f}SE) far {wmean(R.dfar, w):+.4f}"
        print(line, flush=True)
