"""Statistician review (10/07): how much can an 80-window LB difference be trusted?
Per-window IoU on selector-like OOF windows; for model pairs: paired SE inside one virtual LB vs the spread of the
difference across the 39 virtual LBs; cluster structure (panel, panel-day); best_set calibration and robustness.
Read-only: OOF pickles + labels via t2_eval_oof.T. Writes nothing except stdout."""
import sys, os, numpy as np
sys.path.insert(0, r"C:/Users/user/tfb/src")
os.chdir(r"C:/Users/user/tfb/src")
from t2_eval_oof import T, runs, key, load_oof
from t2_model import best_set

SPECS = {"LGB_v10(prior)": "l127_noT_arr", "LGB_v10dem": "l127_noT_arr_dem", "LGBfast_npp": "l127_fast_base",
         "LGBfast_af": "l127_fast_af", "NN_q10": "q10_raw", "NN_q10_nopos": "q10_raw_nopos",
         "NN_af": "q12_af_r13_nopos", "BLEND_af": "l127_noT_arr+q12_af_r13_nopos"}
sel = [k for k in range(T.NWIN) if T.META["is_sel"][k]]


def iou(s, lab, el):
    u = ((s | lab) & el).sum()
    return 1.0 if u == 0 else ((s & lab & el).sum()) / u


IOU, EIOU, P = {}, {}, {}
for nm, spec in SPECS.items():
    oof = load_oof(spec)
    d, e = {}, {}
    for k in sel:
        if k not in oof:
            continue
        o = oof[k]; lab, el = T.LAB[k]
        pe = o.ravel(); order = np.argsort(-pe); cs = np.cumsum(pe[order]); kk = np.arange(1, pe.size + 1)
        ei = cs / (kk + pe.sum() - cs); kb = int(np.argmax(ei)) + 1
        s = np.zeros(pe.size, bool); s[order[:kb]] = True
        d[k] = iou(s.reshape(o.shape), lab, el); e[k] = ei[kb - 1]
    IOU[nm], EIOU[nm] = d, e
    if nm in ("BLEND_af", "LGB_v10(prior)"):
        P[nm] = {k: oof[k] for k in sel if k in oof}
    del oof
    print(f"{nm:16s} n={len(d)} mean IoU {np.mean(list(d.values())):.4f}", flush=True)

# ---------- 1. window-level sd and virtual-LB sampling of differences ----------
vl = {d0: [key[x] for x in ws if x in key] for d0, ws in runs.items()}
vl = {d0: ks for d0, ks in vl.items() if len(ks) >= 60}
print("\n# virtual LBs used:", len(vl), " windows/vLB:", np.mean([len(v) for v in vl.values()]))
b = "BLEND_af"
w_sd = np.mean([np.std([IOU[b][k] for k in ks], ddof=1) for ks in vl.values()])
print(f"window IoU sd inside a vLB ({b}): {w_sd:.3f} -> SE of an 80-window mean {w_sd/np.sqrt(80):.4f} (x0.30 = {0.3*w_sd/np.sqrt(80):.4f} total)")
for cond in ("queue_onset", "queue_ongoing"):
    v = [IOU[b][k] for k in sel if k in IOU[b] and T.META["cond"][k] == cond]
    print(f"  {cond}: mean {np.mean(v):.3f} sd {np.std(v):.3f} share IoU<0.5 {np.mean(np.array(v)<0.5):.3f}")

PAIRS = [("LGB_v10(prior)", "BLEND_af"), ("NN_af", "BLEND_af"), ("LGB_v10(prior)", "LGB_v10dem"),
         ("NN_q10", "NN_q10_nopos"), ("LGBfast_npp", "LGBfast_af"), ("LGB_v10(prior)", "LGBfast_npp")]
print("\n# pair: mean d over vLBs | sd(d) across vLBs | mean paired SE inside a vLB | P(sign agrees with mean) | corr(onset d, next vLB)")
for a, c in PAIRS:
    ds, ses, don = [], [], []
    for d0, ks in sorted(vl.items()):
        ks = [k for k in ks if k in IOU[a] and k in IOU[c]]
        dd = np.array([IOU[c][k] - IOU[a][k] for k in ks])
        ds.append(dd.mean()); ses.append(dd.std(ddof=1) / np.sqrt(len(dd)))
    ds = np.array(ds); m = ds.mean()
    r1 = np.corrcoef(ds[:-1], ds[1:])[0, 1]
    print(f"{a:15s}->{c:15s} {m:+.4f} | {ds.std(ddof=1):.4f} | {np.mean(ses):.4f} | {np.mean(np.sign(ds)==np.sign(m)):.2f} | lag1 r={r1:+.2f}")

# ---------- 2. cluster structure of window IoU (one-way ANOVA ICC) ----------
def icc(groups):
    g = [np.array(v) for v in groups if len(v) >= 2]
    n = np.array([len(v) for v in g]); N = n.sum(); k = len(g)
    grand = np.concatenate(g).mean()
    msb = sum(len(v) * (v.mean() - grand) ** 2 for v in g) / (k - 1)
    msw = sum(((v - v.mean()) ** 2).sum() for v in g) / (N - k)
    n0 = (N - (n ** 2).sum() / N) / (k - 1)
    s2b = max((msb - msw) / n0, 0)
    return s2b / (s2b + msw), n.mean()

allk = sorted({k for ks in vl.values() for k in ks if k in IOU[b]})
for lab_, fn in (("panel", lambda k: T.META["panel"][k]), ("panel x cond", lambda k: (T.META["panel"][k], T.META["cond"][k])),
                 ("panel x month", lambda k: (T.META["panel"][k], T.META["date"][k][:7])),
                 ("panel x day", lambda k: (T.META["panel"][k], T.META["date"][k]))):
    gr = {}
    for k in allk:
        gr.setdefault(fn(k), []).append(IOU[b][k])
    r, m = icc(gr.values())
    print(f"ICC of window IoU by {lab_:14s}: {r:.3f} (mean cluster size {m:.1f})")
# paired-difference ICC by panel (does the advantage of one model concentrate on some panels?)
for a, c in PAIRS[:2]:
    gr = {}
    for k in allk:
        if k in IOU[a]:
            gr.setdefault(T.META["panel"][k], []).append(IOU[c][k] - IOU[a][k])
    r, m = icc(gr.values())
    print(f"ICC of paired difference {a}->{c} by panel: {r:.3f};  per-panel mean d: " +
          " ".join(f"{np.mean(v):+.3f}" for v in gr.values()))

# ---------- 3. best_set: self-assessed expected IoU vs realized ----------
print("\n# calibration of best_set's own expected IoU (selector-like windows)")
for nm in ("BLEND_af", "LGB_v10(prior)", "NN_af"):
    for cond in ("queue_onset", "queue_ongoing"):
        ks = [k for k in sel if k in IOU[nm] and T.META["cond"][k] == cond]
        e = np.array([EIOU[nm][k] for k in ks]); r = np.array([IOU[nm][k] for k in ks])
        bins = np.quantile(e, [0, .25, .5, .75, 1.0])
        z = np.digitize(e, bins[1:-1])
        txt = " ".join(f"[{e[z==i].mean():.2f}->{r[z==i].mean():.2f}]" for i in range(4))
        print(f"{nm:15s} {cond:13s} E[IoU] {e.mean():.3f} realized {r.mean():.3f}  by quartile {txt}")

# ---------- 4. site-rarity strata for onset (true head link vs panel's top-3 onset sites) ----------
from collections import Counter
heads = {}
for k in range(T.NWIN):
    if T.META["cond"][k] == "queue_onset" and T.META["is_eval"][k]:
        y = T.LAB[k][0][-1]
        if y.any():
            heads[k] = int(np.flatnonzero(y).max())
top = {}
for p in set(T.META["panel"]):
    c = Counter(h for k, h in heads.items() if T.META["panel"][k] == p)
    top[p] = [l for l, _ in c.most_common(3)]
def stratum(k):
    y = T.LAB[k][0][-1]
    if not y.any():
        return "none"
    h = int(np.flatnonzero(y).max()); p = T.META["panel"][k]
    if abs(h - top[p][0]) <= 2: return "site1"
    if min(abs(h - t) for t in top[p][1:]) <= 2: return "site2-3"
    return "other"
on_sel = [k for k in sel if T.META["cond"][k] == "queue_onset" and k in IOU[b]]
st = {k: stratum(k) for k in on_sel}
cnt = Counter(st.values()); print("\n# onset strata (selector-like windows):", dict(cnt))
print("model            " + "  ".join(f"{s:>8s}" for s in ("site1", "site2-3", "other")) + "   equal-weight  train-mix")
for nm in SPECS:
    vals = {s: np.mean([IOU[nm][k] for k in on_sel if st[k] == s and k in IOU[nm]]) for s in ("site1", "site2-3", "other")}
    mix = np.mean([IOU[nm][k] for k in on_sel if k in IOU[nm]])
    print(f"{nm:16s} " + "  ".join(f"{vals[s]:8.3f}" for s in vals) + f"   {np.mean(list(vals.values())):.4f}       {mix:.4f}")

# ---------- 5. best_set robustness: logit temperature / shift, and single-segment rule for onset ----------
def lg(p): p = np.clip(p, 1e-6, 1 - 1e-6); return np.log(p / (1 - p))
def sig(z): return 1 / (1 + np.exp(-z))
print("\n# BLEND_af: score under logit scaling a*logit(p)+b before best_set (onset | ongoing)")
for a_ in (0.5, 0.75, 1.0, 1.5, 2.0):
    row = []
    for b_ in (-1.0, 0.0, 1.0):
        res = {}
        for cond in ("queue_onset", "queue_ongoing"):
            ks = [k for k in sel if k in P[b] and T.META["cond"][k] == cond]
            v = []
            for k in ks:
                o = P[b][k]; q = sig(a_ * lg(o) + b_) * (o > 0)
                s = best_set(q.ravel(), np.ones(q.size, bool)).reshape(o.shape)
                v.append(iou(s, *T.LAB[k]))
            res[cond] = np.mean(v)
        row.append(f"b{b_:+.0f}: {res['queue_onset']:.4f}|{res['queue_ongoing']:.4f}")
    print(f"a={a_:4.2f}  " + "   ".join(row))

# single contiguous segment for onset: keep only the run of selected cells that contains the argmax
v0, v1, v2 = [], [], []
for k in on_sel:
    o = P[b][k][-1]; lab, el = T.LAB[k]
    s = best_set(o, np.ones(o.size, bool)); i = int(np.argmax(o))
    seg = np.zeros_like(s); j = i
    while j >= 0 and s[j]: seg[j] = True; j -= 1
    j = i + 1
    while j < len(s) and s[j]: seg[j] = True; j += 1
    S0 = np.zeros((6, o.size), bool); S0[-1] = s
    S1 = np.zeros((6, o.size), bool); S1[-1] = seg
    v0.append(iou(S0, lab, el)); v1.append(iou(S1, lab, el)); v2.append(s.sum() > seg.sum())
v0, v1, v2 = map(np.array, (v0, v1, v2))
print(f"\nonset single-segment rule: best_set {v0.mean():.4f} -> segment {v1.mean():.4f}; windows with >1 segment {v2.mean():.3f}; "
      f"on those: {v0[v2].mean():.3f} -> {v1[v2].mean():.3f}")
for s_ in ("site1", "site2-3", "other"):
    m = np.array([st[k] == s_ for k in on_sel])
    print(f"   {s_:8s}: multi-seg share {v2[m].mean():.3f}, best_set {v0[m].mean():.3f}, segment {v1[m].mean():.3f}")
