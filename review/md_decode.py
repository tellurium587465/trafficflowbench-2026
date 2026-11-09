"""Review (model design): decoding / blending / calibration checks on existing OOF (CPU, read-only).
usage: python md_decode.py
"""
import os, sys, pickle, time, numpy as np
sys.path.insert(0, r"C:/Users/user/tfb/src")
os.environ.setdefault("TFB_T2CACHE", "2_noT")
import t2_eval_oof as E
from t2_model import best_set

t0 = time.time()
M = E.T.META; LAB = E.T.LAB
d = "E:/tfb_nn/nn_data_afdem/"
pn = np.load(d + "panel.npy", allow_pickle=True); tn = np.load(d + "t.npy")
assert (pn == M["panel"]).all() and (tn == M["t"]).all(), "window order differs"
oL = E.oof_of("l127_fast_af"); oN = E.oof_of("q12_af_r13_nopos")
keys = sorted(set(oL) & set(oN))
print("windows", len(keys), f"{time.time()-t0:.0f}s", flush=True)
cond = M["cond"]; panel = M["panel"]
is_on = np.array([cond[k] == "queue_onset" for k in keys])

# virtual-LB runs (selector replica), as in t2_eval_oof
key_sel = E.key
RUNS = []
for d0, ws in E.runs.items():
    ks = [key_sel[x] for x in ws if x in key_sel and key_sel[x] in oL and key_sel[x] in oN]
    if len(ks) >= 60:
        RUNS.append(ks)
SEL = [k for k in keys if M["is_sel"][k]]; EV = [k for k in keys if M["is_eval"][k]]


def iou(s, lab, el):
    u = ((s | lab) & el).sum()
    return 1.0 if u == 0 else ((s & lab & el).sum()) / u


def agg(I, ks):
    from collections import defaultdict
    pc = defaultdict(list)
    for k in ks:
        pc[(panel[k], cond[k])].append(I[k])
    ps = sorted({p for p, _ in pc})
    on = np.mean([np.mean(pc[(p, "queue_onset")]) for p in ps if (p, "queue_onset") in pc])
    og = np.mean([np.mean(pc[(p, "queue_ongoing")]) for p in ps if (p, "queue_ongoing") in pc])
    pm = {p: np.mean([np.mean(pc[(p, c)]) for c in ("queue_onset", "queue_ongoing") if (p, c) in pc]) for p in ps}
    fam = defaultdict(list)
    for p, v in pm.items():
        fam[p.rsplit("_", 1)[0]].append(v)
    return np.mean([np.mean(v) for v in fam.values()]), on, og


def report(name, I):
    v = np.array([agg(I, ks) for ks in RUNS]).mean(0)
    s = agg(I, SEL)
    print(f"{name:34s} vLB {v[0]:.4f} (on {v[1]:.4f} og {v[2]:.4f}) | sel on {s[1]:.4f} og {s[2]:.4f}", flush=True)
    return v


def decode_all(P, fn):
    return {k: iou(fn(P[k], k), LAB[k][0], LAB[k][1]) for k in keys}


base_dec = lambda p, k: best_set(p.ravel(), np.ones(p.size, bool)).reshape(p.shape)
sig = lambda z: 1 / (1 + np.exp(-z)); lg = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / np.clip(1 - p, 1e-6, 1))

# ---------- 0. label structure ----------
on_keys = [k for k in keys if cond[k] == "queue_onset" and M["is_sel"][k]]
early = np.mean([LAB[k][0][:5].any() for k in on_keys])
sz = np.array([LAB[k][0][5].sum() for k in on_keys])
def nseg(r):
    r = r.astype(int); return int(((np.diff(np.concatenate([[0], r])) == 1)).sum())
segs = np.array([nseg(LAB[k][0][5]) for k in on_keys])
print(f"onset(sel) n={len(on_keys)}: label in T+5..T+25 {early:.3f}; cells@T+30 median {np.median(sz)} mean {sz.mean():.2f} "
      f"p90 {np.percentile(sz,90)}; segments: 1={np.mean(segs==1):.3f} 2={np.mean(segs==2):.3f} >=3={np.mean(segs>=3):.3f}")
og_keys = [k for k in keys if cond[k] == "queue_ongoing" and M["is_sel"][k]]
sg = np.array([max(nseg(r) for r in LAB[k][0]) for k in og_keys]); cs = np.array([LAB[k][0].sum() for k in og_keys])
print(f"ongoing(sel) n={len(og_keys)}: max segments/step 1={np.mean(sg==1):.3f} 2={np.mean(sg==2):.3f} >=3={np.mean(sg>=3):.3f}; cells/window median {np.median(cs)}")

# ---------- 1. blend weights (arith vs logit), per condition ----------
res = {}
for w in (0.0, 0.3, 0.5, 0.7, 1.0):
    P = {k: (1 - w) * oL[k] + w * oN[k] for k in keys}
    res[("a", w)] = decode_all(P, base_dec); report(f"arith wNN={w}", res[("a", w)])
for w in (0.5,):
    P = {k: sig((1 - w) * lg(oL[k]) + w * lg(oN[k])) for k in keys}
    for k in keys:
        if cond[k] == "queue_onset": P[k][:5] = 0
    res[("g", w)] = decode_all(P, base_dec); report(f"logit-mean wNN={w}", res[("g", w)])
# per-condition best mix (oracle on vLB: report only)
PB = {k: 0.5 * oL[k] + 0.5 * oN[k] for k in keys}

# ---------- 2. calibration of the 1:1 blend ----------
for nm, sel in (("onset T+30", lambda k: cond[k] == "queue_onset"), ("ongoing", lambda k: cond[k] == "queue_ongoing")):
    pp, yy = [], []
    for k in SEL:
        if not sel(k): continue
        lab, el = LAB[k]; p = PB[k]
        if cond[k] == "queue_onset":
            m = el[5]; pp.append(p[5][m]); yy.append(lab[5][m])
        else:
            pp.append(p[el]); yy.append(lab[el])
    pp = np.concatenate(pp); yy = np.concatenate(yy).astype(float)
    bins = [0, .02, .05, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1.01]
    b = np.digitize(pp, bins) - 1
    row = " ".join(f"[{bins[i]:.2f}+ n={int((b==i).sum())} p={pp[b==i].mean():.3f} y={yy[b==i].mean():.3f}]" for i in range(len(bins) - 1) if (b == i).sum() > 50)
    print(f"calib {nm}: sum p {pp.sum():.0f} vs sum y {yy.sum():.0f}\n  {row}", flush=True)

# ---------- 3. decoders on the 1:1 blend ----------
report("base best_set (1:1)", res[("a", 0.5)])


def shift_dec(a, b, only=None):
    def f(p, k):
        if only and cond[k] != only:
            return base_dec(p, k)
        q = sig(a * lg(p) + b)
        if cond[k] == "queue_onset": q[:5] = 0
        return base_dec(q, k)
    return f


for a, b in ((1, -0.5), (1, 0.5), (0.7, 0), (1.4, 0)):
    report(f"logit a={a} b={b}", decode_all(PB, shift_dec(a, b)))


def kscale(c):
    def f(p, k):
        x = p.ravel(); o = np.argsort(-x); cs_ = np.cumsum(x[o]); tot = x.sum(); kk = np.arange(1, x.size + 1)
        kb = int(np.argmax(cs_ / (kk + tot - cs_))) + 1; kb = max(1, int(round(kb * c)))
        s = np.zeros(x.size, bool); s[o[:kb]] = True; return s.reshape(p.shape)
    return f


for c in (0.8, 1.2):
    report(f"k x{c}", decode_all(PB, kscale(c)))

rng = np.random.default_rng(0)


def mc_dec(S=400):
    def f(p, k):
        x = p.ravel(); o = np.argsort(-x); act = o[x[o] > 1e-3][:400]
        if len(act) == 0:
            return base_dec(p, k)
        pa = x[act]; Z = rng.random((S, len(act))) < pa[None, :]
        tz = Z.sum(1) + (x.sum() - pa.sum())             # expected mass of the rest as a constant
        cz = np.cumsum(Z, axis=1)
        kk = np.arange(1, len(act) + 1)
        e = (cz / np.maximum(kk[None, :] + tz[:, None] - cz, 1e-9)).mean(0)
        kb = int(np.argmax(e)) + 1
        s = np.zeros(x.size, bool); s[act[:kb]] = True; return s.reshape(p.shape)
    return f


report("exact E[IoU] (MC, independence)", decode_all(PB, mc_dec()))


def onset_comp(frac=0.1, gap=1):
    """onset: keep only the connected component (in link order, gaps<=gap) with the largest mass at T+30."""
    def f(p, k):
        if cond[k] != "queue_onset":
            return base_dec(p, k)
        r = p[5]; m = r >= frac * r.max(); idx = np.flatnonzero(m)
        comps, cur = [], [idx[0]]
        for i in idx[1:]:
            if i - cur[-1] <= gap + 1: cur.append(i)
            else: comps.append(cur); cur = [i]
        comps.append(cur)
        best = max(comps, key=lambda c: r[c[0]:c[-1] + 1].sum())
        q = np.zeros_like(p); q[5, best[0]:best[-1] + 1] = r[best[0]:best[-1] + 1]
        return base_dec(q, k)
    return f


for fr in (0.05, 0.2):
    report(f"onset single component frac={fr}", decode_all(PB, onset_comp(fr)))


def fill_gaps(g=1):
    def f(p, k):
        s = base_dec(p, k)
        for r in range(s.shape[0]):
            idx = np.flatnonzero(s[r])
            for a_, b_ in zip(idx[:-1], idx[1:]):
                if 1 < b_ - a_ <= g + 1: s[r, a_:b_] = True
        return s
    return f


report("fill 1-link gaps per step", decode_all(PB, fill_gaps(1)))

# ---------- 4. onset diagnostics: model hit@1 vs location-free heuristics ----------
nm = list(np.load(d + "names.npy", allow_pickle=True)); X = np.load(d + "X.npy", mmap_mode="r"); OFF = np.load(d + "off.npy")
cols = ["last_v", "v_min", "v_m3", "v_trend", "q_m3", "dem30_over_dnmin", "arr6", "capl"]
ci = [nm.index(c) for c in cols]
hits = {c: [] for c in cols}; hits["model"] = []; hits["model_rank"] = []
for k in on_keys:
    lab = LAB[k][0][5]; Xi = np.asarray(X[OFF[k]:OFF[k + 1]])[:, ci]
    tr = np.flatnonzero(lab)
    hits["model"].append(lab[np.argmax(PB[k][5])])
    rk = (-PB[k][5]).argsort().argsort(); hits["model_rank"].append(rk[tr].min())
    for j, c in enumerate(cols):
        v = np.where(np.isfinite(Xi[:, j]), Xi[:, j], np.nan)
        if np.all(np.isnan(v)): hits[c].append(False); continue
        sgn = 1 if c in ("last_v", "v_min", "v_m3", "v_trend", "capl") else -1   # small speed / big demand ratio = risky
        hits[c].append(lab[np.nanargmin(sgn * v)])
print("onset(sel) hit@1 (argmax link in true set):", {c: round(float(np.mean(v)), 3) for c, v in hits.items() if c != "model_rank"})
mr = np.array(hits["model_rank"]); print(f"  model best rank of a true link: <=1 {np.mean(mr<1):.3f} <=3 {np.mean(mr<3):.3f} <=5 {np.mean(mr<5):.3f} <=10 {np.mean(mr<10):.3f}")
# same for LOPO NN (location memory impossible)
oP = E.oof_of("q6_lopo_dem")
h = [LAB[k][0][5][np.argmax(oP[k][5])] for k in on_keys if k in oP]
print(f"LOPO NN (q6_lopo_dem) onset hit@1 {np.mean(h):.3f}; max p median {np.median([oP[k][5].max() for k in on_keys if k in oP]):.3f}")
print("done", f"{time.time()-t0:.0f}s")
