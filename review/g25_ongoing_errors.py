"""G25: where does the ongoing IoU go? s52-like pipeline on the OOF (no-memory blend, OGB 0.85 evidence, far gate),
official-like windows (exact persistence IoU <= 0.9). Per window: false negatives / false positives by region relative
to the last visible queue (core = queued in the last history row, edge = within 3 links, far = beyond), by horizon
step, and by window type (truth at T+30 vs the last visible queue: grow > 1.25x, shrink < 0.8x, else steady).
Loss share = (FN + FP) of a category / total (union - intersection). Persistence (last row repeated) for reference."""
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


rows, steps = [], []
for k in O.SEL:
    if k not in oof or per[k] > 0.9:
        continue
    p = str(T.META["panel"][k]); t = int(T.META["t"][k]); day = pd.Timestamp(str(T.META["date"][k])[:10]); q = qobs(p)
    seen = np.zeros(q.shape[1], bool); nd = 0
    for b in (1, 2, 3):
        dd = t - b * 288
        if dd - 18 >= 0 and (day - pd.Timedelta(days=b)).month == day.month:
            seen |= q[dd - 18:dd + 12].any(axis=0); nd += 1
    d0 = t - (t % 288) + 60; gs = seen | (q[d0:t].any(axis=0) if t > d0 else False); gev = gs.copy()
    for o in (1, 2):
        gev[o:] |= gs[:-o]; gev[:-o] |= gs[o:]
    pr = oof[k].copy()
    if nd:
        ev = seen.copy(); ev[1:] |= seen[:-1]; ev[:-1] |= seen[1:]; pr = pr * np.where(ev, 1.0, 0.85)[None, :]
    S = bestset(pr) & ~(far_mask(k) & ~gev)[None, :]
    lab, el = T.LAB[k]; Y = lab.astype(bool) & el.astype(bool); S = S & el.astype(bool)
    lq = _FARM["lq"][_FARM["off"][k]:_FARM["off"][k + 1]]; n = len(lq)
    dist = np.full(n, 99)
    for j in np.flatnonzero(lq):
        dist = np.minimum(dist, np.abs(np.arange(n) - j))
    reg = np.where(lq, "core", np.where(dist <= 3, "edge", "far"))
    Pm = np.broadcast_to(lq, Y.shape) & el.astype(bool)
    y30, l0 = Y[-1].sum(), max(lq.sum(), 1)
    typ = "grow" if y30 > 1.25 * l0 else ("shrink" if y30 < 0.8 * l0 else "steady")
    u = (S | Y).sum(); i_ = (S & Y).sum(); up = (Pm | Y).sum(); ip = (Pm & Y).sum()
    r = dict(typ=typ, iou=1.0 if u == 0 else i_ / u, iou_pers=1.0 if up == 0 else ip / up, loss=u - i_, union=u)
    for g in ("core", "edge", "far"):
        m = (reg == g)[None, :]
        r[f"FN_{g}"] = (Y & ~S & m).sum(); r[f"FP_{g}"] = (S & ~Y & m).sum()
    rows.append(r)
    for s_ in range(Y.shape[0]):
        steps.append(dict(step=5 * (s_ + 1), FN=(Y[s_] & ~S[s_]).sum(), FP=(S[s_] & ~Y[s_]).sum(), U=(S[s_] | Y[s_]).sum()))
R = pd.DataFrame(rows); St = pd.DataFrame(steps)
print(f"official-like ongoing windows {len(R)}: IoU {R.iou.mean():.4f} (persistence {R.iou_pers.mean():.4f})")
tot = R.loss.sum()
print("loss share by region:", {c: round(R[c].sum() / tot, 3) for c in R.columns if c.startswith(("FN_", "FP_"))})
g = R.groupby("typ").agg(n=("iou", "size"), iou=("iou", "mean"), pers=("iou_pers", "mean"), loss=("loss", "sum"))
g["loss_share"] = g.loss / tot; print(g.round(3))
for t_ in ("grow", "shrink", "steady"):
    x = R[R.typ == t_]; print(t_, {c: round(x[c].sum() / max(x.loss.sum(), 1), 3) for c in R.columns if c.startswith(("FN_", "FP_"))})
st = St.groupby("step")[["FN", "FP", "U"]].sum(); st["miss_rate"] = st.FN / st.U; st["false_rate"] = st.FP / st.U; print(st.round(3))
