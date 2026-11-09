"""T2 ongoing yardstick for the test-month shift (10/08 night). On the LB the persistence baseline scores the same as
locally (0.603 vs 0.59-0.60) but the model falls from ~0.94 to ~0.82, i.e. the model loses what it learned about where
queues grow. Same construction as the onset yardstick (t2_eval_surprise.py), on the selector-like ongoing windows:
f = frequency of the window's queue site (3-link bucket of the queue head at the horizon end) given
(panel, weekend, 2-h bin), leaving the window's month out. Reports plain, 1/max(f,0.05)-weighted and f<0.3 IoU, with
clustered (panel x date) bootstrap SE of differences vs the first spec.

usage: python t2_eval_ongoing_shift.py <spec_ref> <spec2> [...]
"""
import sys, numpy as np
from collections import Counter, defaultdict
from t2_eval_oof import T, load_oof, far_mask
from t2_eval_surprise import _head, _slot

C = "queue_ongoing"
SEL = [k for k in range(T.NWIN) if T.META["is_sel"][k] and T.META["cond"][k] == C]
EV = [k for k in range(T.NWIN) if (T.META["is_eval"][k] or T.META["is_sel"][k]) and T.META["cond"][k] == C]
_cnt = defaultdict(Counter)
for k in EV:
    _cnt[(_slot(k), str(T.META["date"][k])[:7])][_head(k)] += 1


def _f(k):
    s, m, h = _slot(k), str(T.META["date"][k])[:7], _head(k)
    num = sum(c[h] for (ss, mm), c in _cnt.items() if ss == s and mm != m)
    den = sum(sum(c.values()) for (ss, mm), c in _cnt.items() if ss == s and mm != m)
    return (num + 0.5) / (den + 1.0)


F = np.array([_f(k) for k in SEL]); W1 = 1 / np.maximum(F, 0.05); SUB = F < 0.3
CL = np.array([f"{T.META['panel'][k]}|{str(T.META['date'][k])[:10]}" for k in SEL])


def _iou_bs(o, lab, el, far=None):                         # best_set, then the TFB_T2_FAR rule (far is None when off)
    pe = o.ravel(); order = np.argsort(-pe); cs = np.cumsum(pe[order]); kk = np.arange(1, pe.size + 1)
    kb = int(np.argmax(cs / (kk + pe.sum() - cs))) + 1; s = np.zeros(pe.size, bool); s[order[:kb]] = True
    s = s.reshape(o.shape)
    if far is not None:
        s = s & ~far[None, :]
    lab = lab.astype(bool); el = el.astype(bool); u = ((s | lab) & el).sum()
    return 1.0 if u == 0 else (s & lab & el).sum() / u


def per_window(spec):
    o = load_oof(spec)
    return np.array([_iou_bs(o[k], *T.LAB[k], far_mask(k)) if k in o else np.nan for k in SEL])


def stats(r):
    ok = np.isfinite(r)
    return (r[ok].mean(), np.sum(W1[ok] * r[ok]) / W1[ok].sum(), r[ok & SUB].mean())


def boot_se(d, reps=400, seed=0):
    rng = np.random.default_rng(seed); u = np.unique(CL); idx = {c: np.flatnonzero(CL == c) for c in u}; out = []
    for _ in range(reps):
        ii = np.concatenate([idx[c] for c in rng.choice(u, len(u))]); dd = d[ii]; ok = np.isfinite(dd)
        out.append((dd[ok].mean(), np.sum(W1[ii][ok] * dd[ok]) / W1[ii][ok].sum(), dd[ok & SUB[ii]].mean()))
    return np.std(out, axis=0)


if __name__ == "__main__":
    specs = sys.argv[1:]; R = {s: per_window(s) for s in specs}
    print(f"ongoing windows {len(SEL)}, f<0.3 subset {SUB.sum()}, ESS(gamma=1) {W1.sum()**2/np.sum(W1**2):.0f}, "
          f"f quartiles {np.round(np.quantile(F, [0.25, 0.5, 0.75]), 2)}")
    print(f"{'spec':40s} plain    weighted  f<0.3")
    ref = R[specs[0]]
    for s in specs:
        a = stats(R[s]); line = f"{s[:40]:40s} {a[0]:.4f}   {a[1]:.4f}    {a[2]:.4f}"
        if s != specs[0]:
            d = R[s] - ref; m = stats(d); se = boot_se(d)
            line += "   | diff " + "  ".join(f"{x:+.4f}({x/max(e,1e-9):+.1f}SE)" for x, e in zip(m, se))
        print(line, flush=True)
