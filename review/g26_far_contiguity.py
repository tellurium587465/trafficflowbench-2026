"""G26: far (> 3 links from the last visible queue) ongoing cells - contiguous growth or detached new queues?
For each official-like ongoing window and step: truth cells beyond 3 links that belong to a truth run (connected in
link order within the step) containing a core/edge cell = 'contiguous growth', else 'detached'. Same split for the
model's far positives BEFORE the gate (raw expected-IoU set) and their precision; and what the s40 gate removes of each.
Links are in traffic order (F19), so upstream = lower index."""
import sys, os, numpy as np, pandas as pd
sys.path.insert(0, r"C:/Users/user/tfb/src")
os.environ.setdefault("TFB_T2_FAR", "3")
import t2_eval_ongoing_shift as O
from t2_eval_oof import T, load_oof, far_mask, _FARM
from cachelib import load

SPEC = "q25_nomem_s3+q25_nomem_s4+q31_nomem_wp05_s0:2+q31_nomem_attn1_s0:2+q31_nomem_r13_s0+q31_nomem_drop1_s0"
oof = load_oof(SPEC); per = np.load("E:/tfb_nn/persist_iou_afdem.npy"); PC = {}


def qobs(p):
    if p not in PC:
        P = load(p, "train"); q = (P.V <= P.vcut[None, :]) & np.isfinite(P.V)
        PC[p] = q & (np.random.default_rng(0).random(q.shape) > 0.435)
    return PC[p]


def bestset(p):
    pe = p.ravel(); o = np.argsort(-pe); cs = np.cumsum(pe[o]); kk = np.arange(1, pe.size + 1)
    kb = int(np.argmax(cs / (kk + pe.sum() - cs))) + 1; s = np.zeros(pe.size, bool); s[o[:kb]] = True
    return s.reshape(p.shape)


def contiguous(row, near):
    """cells of the boolean row whose run (consecutive links) touches a near link"""
    out = np.zeros_like(row); n = len(row); i = 0
    while i < n:
        if row[i]:
            j = i
            while j + 1 < n and row[j + 1]:
                j += 1
            if near[i:j + 1].any() or (i > 0 and near[i - 1]) or (j + 1 < n and near[j + 1]):
                out[i:j + 1] = True
            i = j + 1
        else:
            i += 1
    return out


acc = dict(y_far=0, y_far_contig=0, s_far=0, s_far_contig=0, s_far_contig_hit=0, s_far_det=0, s_far_det_hit=0,
           gate_rm_contig=0, gate_rm_contig_hit=0, gate_rm_det=0, gate_rm_det_hit=0, upstream=0, downstream=0)
for k in O.SEL:
    if k not in oof or per[k] > 0.9:
        continue
    p = str(T.META["panel"][k]); t = int(T.META["t"][k]); day = pd.Timestamp(str(T.META["date"][k])[:10]); q = qobs(p)
    seen = np.zeros(q.shape[1], bool)
    for b in (1, 2, 3):
        dd = t - b * 288
        if dd - 18 >= 0 and (day - pd.Timedelta(days=b)).month == day.month:
            seen |= q[dd - 18:dd + 12].any(axis=0)
    d0 = t - (t % 288) + 60; gs = seen | (q[d0:t].any(axis=0) if t > d0 else False); gev = gs.copy()
    for o in (1, 2):
        gev[o:] |= gs[:-o]; gev[:-o] |= gs[o:]
    lab, el = T.LAB[k]; el = el.astype(bool); Y = lab.astype(bool) & el
    fm = far_mask(k); near = ~fm; lq = _FARM["lq"][_FARM["off"][k]:_FARM["off"][k + 1]]; n = len(lq)
    S = bestset(oof[k]) & el; gate = fm & ~gev
    if lq.any():
        lo, hi = np.flatnonzero(lq).min(), np.flatnonzero(lq).max()
    for s_ in range(Y.shape[0]):
        cy = contiguous(Y[s_], near); cs_ = contiguous(S[s_], near)
        yf = Y[s_] & fm; sf = S[s_] & fm
        acc["y_far"] += yf.sum(); acc["y_far_contig"] += (yf & cy).sum()
        if lq.any():
            acc["upstream"] += (yf & (np.arange(n) < lo)).sum(); acc["downstream"] += (yf & (np.arange(n) > hi)).sum()
        acc["s_far"] += sf.sum()
        acc["s_far_contig"] += (sf & cs_).sum(); acc["s_far_contig_hit"] += (sf & cs_ & Y[s_]).sum()
        acc["s_far_det"] += (sf & ~cs_).sum(); acc["s_far_det_hit"] += (sf & ~cs_ & Y[s_]).sum()
        rm = sf & gate
        acc["gate_rm_contig"] += (rm & cs_).sum(); acc["gate_rm_contig_hit"] += (rm & cs_ & Y[s_]).sum()
        acc["gate_rm_det"] += (rm & ~cs_).sum(); acc["gate_rm_det_hit"] += (rm & ~cs_ & Y[s_]).sum()
a = acc
print(f"truth far cells {a['y_far']}: contiguous with the visible queue {a['y_far_contig']/a['y_far']:.3f}; upstream of it {a['upstream']/a['y_far']:.3f}, downstream {a['downstream']/a['y_far']:.3f}")
print(f"model far positives {a['s_far']}: contiguous {a['s_far_contig']} (precision {a['s_far_contig_hit']/max(a['s_far_contig'],1):.3f}), detached {a['s_far_det']} (precision {a['s_far_det_hit']/max(a['s_far_det'],1):.3f})")
print(f"gate removes: contiguous {a['gate_rm_contig']} (of which right {a['gate_rm_contig_hit']}), detached {a['gate_rm_det']} (right {a['gate_rm_det_hit']})")
