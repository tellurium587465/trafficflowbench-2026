"""Review: (a) site-prior correction for onset under a simulated site-mix shift, (b) LB-delta noise of paired
comparisons on LB-sized samples, (c) exact E[IoU] decode, (d) location-free onset heuristics vs model / LOPO NN."""
import os, sys, time, numpy as np
from collections import defaultdict
sys.path.insert(0, r"C:/Users/user/tfb/src")
os.environ.setdefault("TFB_T2CACHE", "2_noT")
import t2_eval_oof as E
from t2_model import best_set, FOLDS

t0 = time.time()
M = E.T.META; LAB = E.T.LAB; cond = M["cond"]; panel = M["panel"]
oL = E.oof_of("l127_fast_af"); oN = E.oof_of("q12_af_r13_nopos")
keys = sorted(set(oL) & set(oN))
PB = {k: 0.5 * oL[k] + 0.5 * oN[k] for k in keys}
RUNS = []
for d0, ws in E.runs.items():
    ks = [E.key[x] for x in ws if x in E.key and E.key[x] in PB]
    if len(ks) >= 60: RUNS.append(ks)
mon = np.array([d[:7] for d in M["date"]]); fold_of = {m: i for i, f in enumerate(FOLDS) for m in f}


def iou(s, lab, el):
    u = ((s | lab) & el).sum()
    return 1.0 if u == 0 else ((s & lab & el).sum()) / u


def dec(p):
    return best_set(p.ravel(), np.ones(p.size, bool)).reshape(p.shape)


def agg(I, ks, w=None):
    pc = defaultdict(list); pw = defaultdict(list)
    for k in ks:
        pc[(panel[k], cond[k])].append(I[k]); pw[(panel[k], cond[k])].append(1.0 if w is None else w.get(k, 1.0))
    ps = sorted({p for p, _ in pc})
    m = lambda key: np.average(pc[key], weights=pw[key])
    on = np.mean([m((p, "queue_onset")) for p in ps if (p, "queue_onset") in pc])
    og = np.mean([m((p, "queue_ongoing")) for p in ps if (p, "queue_ongoing") in pc])
    return on, og


# ---- (b) LB-delta noise: sd over LB-sized runs of paired differences ----
I_L = {k: iou(dec(oL[k]), *LAB[k]) for k in keys}; I_N = {k: iou(dec(oN[k]), *LAB[k]) for k in keys}
I_B = {k: iou(dec(PB[k]), *LAB[k]) for k in keys}
for nm, A, B in (("blend - LGBM", I_B, I_L), ("blend - NN", I_B, I_N)):
    d = np.array([np.subtract(agg(A, ks), agg(B, ks)) for ks in RUNS])
    print(f"paired diff {nm}: onset mean {d[:,0].mean():+.4f} sd {d[:,0].std():.4f} | ongoing mean {d[:,1].mean():+.4f} sd {d[:,1].std():.4f}  (runs {len(d)})")
a = np.array([agg(I_B, ks) for ks in RUNS]); print(f"blend level across LB-sized runs: onset sd {a[:,0].std():.4f} ongoing sd {a[:,1].std():.4f}")

# ---- (a) site prior: per panel, frequency of each link queued at T+30 in onset windows of the OTHER folds ----
onk = [k for k in keys if cond[k] == "queue_onset"]
allon = [k for k in range(E.T.NWIN) if cond[k] == "queue_onset"]
prior = {}
for f in range(3):
    acc = {}
    for k in allon:
        if fold_of.get(mon[k], -1) == f: continue
        lab = LAB[k][0][5].astype(float)
        a_ = acc.setdefault(panel[k], [np.zeros_like(lab), 0]); a_[0] += lab; a_[1] += 1
    for p, (s, n) in acc.items():
        fr = s / n; fr = np.convolve(fr, np.ones(3) / 3, "same")
        prior[(f, p)] = fr
def head(k):
    q = np.flatnonzero(LAB[k][0][5]); return q.max() if len(q) else -1
# shift simulation weight: 1 / prior(head)^beta (flattens the site mix inside each panel)
def wts(beta):
    w = {}
    for k in onk:
        h = head(k); pr = prior[(fold_of[mon[k]], panel[k])]
        w[k] = 1.0 / max(pr[h] if h >= 0 else 1.0, 0.01) ** beta
    return w
W5, W10 = wts(0.5), wts(1.0)
eff = lambda w: (sum(w.values()) ** 2) / sum(v * v for v in w.values())
print(f"onset windows {len(onk)}; effective n under shift weights beta .5: {eff(W5):.0f}, beta 1: {eff(W10):.0f}")


def corr(alpha, P=PB):
    out = {}
    for k in keys:
        p = P[k]
        if cond[k] != "queue_onset" or alpha == 0:
            out[k] = p; continue
        pr = prior[(fold_of[mon[k]], panel[k])] + 1e-3
        tgt = pr ** (1 - alpha) * pr.mean() ** alpha                    # tempered toward a flat site prior
        r1, r0 = tgt / pr, (1 - tgt) / np.maximum(1 - pr, 1e-3)
        q = p.copy(); x = p[5]; q[5] = r1 * x / (r1 * x + r0 * (1 - x) + 1e-12)   # Saerens et al. 2002 prior correction
        out[k] = q
    return out


for alpha in (0.0, 0.25, 0.5, 1.0):
    Pc = corr(alpha); I = {k: iou(dec(Pc[k]), *LAB[k]) for k in onk}
    base = np.mean([agg(I, [k for k in ks if cond[k] == "queue_onset"])[0] for ks in RUNS])
    print(f"prior-correction alpha={alpha}: onset vLB {base:.4f} | all onset windows: plain {agg(I, onk)[0]:.4f} "
          f"shift beta.5 {agg(I, onk, W5)[0]:.4f} shift beta1 {agg(I, onk, W10)[0]:.4f}", flush=True)
for nm, P in (("LGBM", oL), ("NN", oN)):
    I = {k: iou(dec(P[k]), *LAB[k]) for k in onk}
    print(f"  {nm} alone: plain {agg(I, onk)[0]:.4f} shift beta.5 {agg(I, onk, W5)[0]:.4f} shift beta1 {agg(I, onk, W10)[0]:.4f}")

# ---- (c) exact E[IoU] under independence (MC, no cap) ----
rng = np.random.default_rng(0)


def mc(p, S=300):
    x = p.ravel(); o = np.argsort(-x); act = o[x[o] > 1e-3]
    if len(act) == 0: return dec(p)
    pa = x[act]; Z = rng.random((S, len(act))) < pa[None, :]
    tz = Z.sum(1) + (x.sum() - pa.sum()); cz = np.cumsum(Z, axis=1); kk = np.arange(1, len(act) + 1)
    e = (cz / np.maximum(kk[None, :] + tz[:, None] - cz, 1e-9)).mean(0)
    s = np.zeros(x.size, bool); s[act[:int(np.argmax(e)) + 1]] = True; return s.reshape(p.shape)


I_mc = {k: iou(mc(PB[k]), *LAB[k]) for k in keys}
v0 = np.array([agg(I_B, ks) for ks in RUNS]).mean(0); v1 = np.array([agg(I_mc, ks) for ks in RUNS]).mean(0)
print(f"decode ratio-of-expectations: on {v0[0]:.4f} og {v0[1]:.4f} | exact E[IoU] MC: on {v1[0]:.4f} og {v1[1]:.4f}")

# ---- (d) onset hit@1: model vs location-free heuristics vs LOPO NN ----
d = "E:/tfb_nn/nn_data_afdem/"
nm = list(np.load(d + "names.npy", allow_pickle=True)); X = np.load(d + "X.npy", mmap_mode="r"); OFF = np.load(d + "off.npy")
cols = ["last_v", "v_min", "v_m3", "dem30_over_dnmin", "arr6"]; ci = [nm.index(c) for c in cols]
oP = E.oof_of("q6_lopo_dem")
H = defaultdict(list)
sel_on = [k for k in onk if M["is_sel"][k] and LAB[k][0][5].any()]
for k in sel_on:
    lab = LAB[k][0][5]; Xi = np.asarray(X[OFF[k]:OFF[k + 1]])[:, ci]
    H["blend"].append(lab[np.argmax(PB[k][5])]); H["LGBM"].append(lab[np.argmax(oL[k][5])]); H["NN"].append(lab[np.argmax(oN[k][5])])
    if k in oP: H["NN_LOPO"].append(lab[np.argmax(oP[k][5])])
    H["random"].append(lab.mean())
    for j, c in enumerate(cols):
        v = Xi[:, j].astype(float)
        if np.all(~np.isfinite(v)): H[c].append(False); continue
        v = np.where(np.isfinite(v), v, np.nan)
        H[c].append(lab[np.nanargmin(v) if c in ("last_v", "v_min", "v_m3") else np.nanargmax(v)])
print("onset(sel) hit@1:", {c: round(float(np.mean(v)), 3) for c, v in H.items()})
print("done", f"{time.time()-t0:.0f}s")
