"""T2 onset yardstick for the test-month shift (statistics review 10/07, proposal 1).

Train onset windows of the selector-like set, weighted by how atypical their true site is for the calendar slot:
f = frequency of the true site (3-link bucket) given (panel, weekend, 2-h time bin), leaving the window's month out.
Reports per spec: plain onset IoU, weight 1/max(f,0.05) (gamma=1), and the f<0.3 subset (17% of windows; reproduced
the LB level 0.65-0.75 and the val-month overconfidence). Differences vs the first spec get a clustered bootstrap SE
(clusters = panel x date), so a change is only called when |diff| > 2 SE.

usage: python t2_eval_surprise.py <spec_ref> <spec2> [...]      (spec syntax as t2_eval_oof: "a:2+b")
"""
import sys, numpy as np, pandas as pd
from collections import Counter, defaultdict
from t2_eval_oof import T, load_oof

SEL = [k for k in range(T.NWIN) if T.META["is_sel"][k] and T.META["cond"][k] == "queue_onset"]
EV = [k for k in range(T.NWIN) if (T.META["is_eval"][k] or T.META["is_sel"][k]) and T.META["cond"][k] == "queue_onset"]


def _head(k):
    y = T.LAB[k][0][-1]
    return int(np.flatnonzero(y).max()) // 3 if y.any() else -1


def _slot(k):
    return (T.META["panel"][k], pd.Timestamp(str(T.META["date"][k])).dayofweek >= 5, (int(T.META["t"][k]) % 288) // 24)


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


def _iou_bs(o, lab, el):
    pe = o.ravel(); order = np.argsort(-pe); cs = np.cumsum(pe[order]); kk = np.arange(1, pe.size + 1)
    ei = cs / (kk + pe.sum() - cs); kb = int(np.argmax(ei)) + 1; s = np.zeros(pe.size, bool); s[order[:kb]] = True
    s = s.reshape(o.shape); lab = lab.astype(bool); el = el.astype(bool); u = ((s | lab) & el).sum()
    return 1.0 if u == 0 else (s & lab & el).sum() / u


def per_window(spec):
    o = load_oof(spec)
    return np.array([_iou_bs(o[k], *T.LAB[k]) if k in o else np.nan for k in SEL])


def stats(r):
    ok = np.isfinite(r)
    return (r[ok].mean(), np.sum(W1[ok] * r[ok]) / W1[ok].sum(), r[ok & SUB].mean())


def boot_se(d, reps=400, seed=0):
    """clustered bootstrap SE of (plain, weighted, subset) means of a per-window difference d."""
    rng = np.random.default_rng(seed); u = np.unique(CL); idx = {c: np.flatnonzero(CL == c) for c in u}; out = []
    for _ in range(reps):
        ii = np.concatenate([idx[c] for c in rng.choice(u, len(u))]); dd = d[ii]; ok = np.isfinite(dd)
        out.append((dd[ok].mean(), np.sum(W1[ii][ok] * dd[ok]) / W1[ii][ok].sum(), dd[ok & SUB[ii]].mean()))
    return np.std(out, axis=0)


if __name__ == "__main__":
    specs = sys.argv[1:]; R = {s: per_window(s) for s in specs}
    print(f"onset windows {len(SEL)}, f<0.3 subset {SUB.sum()}, ESS(gamma=1) {W1.sum()**2/np.sum(W1**2):.0f}")
    print(f"{'spec':40s} plain    weighted  f<0.3")
    ref = R[specs[0]]
    for s in specs:
        a = stats(R[s]); line = f"{s:40s} {a[0]:.4f}   {a[1]:.4f}    {a[2]:.4f}"
        if s != specs[0]:
            d = R[s] - ref; m = stats(d); se = boot_se(d)
            line += "   | diff " + "  ".join(f"{x:+.4f}({x/max(e,1e-9):+.1f}SE)" for x, e in zip(m, se))
        print(line, flush=True)
