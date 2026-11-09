"""G336 (11/05, ledger #269, measurement only): the likelihood ratio a new piece of evidence would need to flip the onset choice. On the T+30
row just before best_set (the k draw of largest weight): blobs = maximal runs of links with p > 0.05 (as #235); the chosen blob = the blob
with the largest mass among those best_set selects from; the true location = the true scored cells +-2 links, its mass = p summed there.
Misses (IoU < 0.5): chosen mass / true-location mass. Hits: chosen mass / the mass of the largest other blob (the margin a wrong flip
would need). The month map of #266 (other days' head onsets within +-1 h of the T+30 clock, oracle) as a likelihood ratio: map rate at
the true location / at the chosen blob (+0.5 day pseudo-count). Instruments: M3 dev (A17 onset rows, G298 M), January / February (capB0
rows, G298 J), and April 40 (A16 = A17's onset, G289: chosen / second only). usage: TFB_M3VER=v3 py -B -X utf8 g336_flip_ratio.py"""
import os, sys, pickle, importlib.util
os.environ.setdefault("TFB_M3VER", "v3")
import numpy as np, pandas as pd
R0 = r"C:/Users/user/tfb/"; sys.path.insert(0, r"C:/Users/user/tfb_a8/src"); sys.path.insert(0, R0 + "src")
from cachelib import load, T2_PANELS
import t2_m3_gate as G
from t2_model import best_set
W = R0 + "work/t2/"; dil = lambda m, r: np.convolve(m.astype(int), np.ones(2 * r + 1, int), mode="same") > 0


def blobs(row, th=0.05):
    out, j = [], 0
    while j < len(row):
        if row[j] > th:
            e_ = j
            while e_ + 1 < len(row) and row[e_ + 1] > th:
                e_ += 1
            out.append((j, e_)); j = e_ + 1
        else:
            j += 1
    return out


def heads(V, vcut):
    fin = np.isfinite(V); qd = fin & (V <= vcut[None, :]); fr = fin & ~qd; free12 = np.zeros_like(fr); run = np.zeros(V.shape[1], int)
    for r in range(len(V)):
        free12[r] = run >= 12; run = np.where(fr[r], run + 1, np.where(fin[r], 0, run))
    return qd & free12 & ~np.concatenate([qd[:, 1:], np.zeros((len(qd), 1), bool)], axis=1)


def map_rate(H, day, tod):
    nd = H.shape[0] // 288; other = [d for d in range(nd) if d != day]; Hd = H[:nd * 288].reshape(nd, 288, -1)[other]; lo, hi = max(tod - 12, 0), min(tod + 13, 288)
    return (Hd[:, lo:hi].any(axis=1).sum(axis=0) + 0.5) / (len(other) + 1)


def one(row, y, e, mr):
    yt = y & e; sel = best_set(row, np.ones(row.size, bool)); iou = 1.0 if ((sel | y) & e).sum() == 0 else float((sel & y & e).sum() / ((sel | y) & e).sum())
    bl = blobs(row); ms = [float(row[a:b + 1].sum()) for a, b in bl]; ch = [i for i, (a, b) in enumerate(bl) if sel[a:b + 1].any()]
    if not ch:
        return None
    ci = max(ch, key=lambda i: ms[i]); a, b = bl[ci]; cm = ms[ci]; oth = [ms[i] for i in range(len(bl)) if i != ci]; tl = dil(yt, 2)
    out = dict(iou=iou, chosen=cm, second=max(oth) if oth else 0.0, true_mass=float(row[tl].sum()) if yt.any() else np.nan)
    if mr is not None and yt.any():
        out["map_lr"] = float(mr[tl].max() / max(mr[a:b + 1].max(), 1e-9))
    return out


def summary(D, lab):
    mi = D[D.iou < 0.5]; hi = D[D.iou >= 0.5]; q = lambda v: " / ".join(f"{x:.2f}" for x in np.nanquantile(v, [0.1, 0.25, 0.5, 0.75, 0.9])) if len(v) else "-"
    print(f"--- {lab}: {len(D)} windows, misses {len(mi)} | misses: chosen / true-location mass (p10/25/50/75/90) {q(mi.chosen / mi.true_mass)}"
          + (f" | map LR true / chosen {q(mi.map_lr)}" if "map_lr" in D and len(mi) else "") + f" | hits: chosen / second {q(hi.chosen / np.maximum(hi.second, 1e-3))}"
          + (f" | hits map LR {q(hi.map_lr)}" if "map_lr" in D else ""), flush=True)


VM = {}
for p in T2_PANELS:
    P = load(p, "validation"); VM[p] = heads(P.V, P.vcut); load.cache_clear()
D8 = pickle.load(open(W + "g298_on_pretemp.pkl", "rb")); rows = []
for w, v in D8["M"].items():
    m = G.META[w]; pr, row = max(v["A"], key=lambda x: x[0]); r = one(row, v["y"], v["e"], map_rate(VM[m["panel"]], m["day"], (m["t"] + 6) % 288))
    if r:
        rows.append(r)
summary(pd.DataFrame(rows), "M3 dev onset (A17)")
spec = importlib.util.spec_from_file_location("g119", R0 + "review/g119_month_postproc.py"); h = importlib.util.module_from_spec(spec); sys.argv = ["g119", "run"]; spec.loader.exec_module(h); T_ = h.T; MON = h.MON
HT = {}; rows = []
for k, v in D8["J"].items():
    p = str(T_.META["panel"][k]); t = int(T_.META["t"][k]); mo = MON[k]
    if (p, mo) not in HT:
        P = h.load(p, "train"); rr = np.flatnonzero([s_.startswith(mo) for s_ in P.times]); HT[(p, mo)] = (heads(np.where(P.blank, np.nan, P.V)[rr], P.vcut), int(rr[0])); h.load.cache_clear()
    H, r0 = HT[(p, mo)]; pr, row = max(v["C"], key=lambda x: x[0]); r = one(row, v["y"], v["e"], map_rate(H, (t - r0) // 288, (t + 6) % 288))
    if r:
        rows.append(r)
summary(pd.DataFrame(rows), "January / February onset (capB0)")
A = pickle.load(open(W + "g289_april.pkl", "rb")); ra = []
for w, v in A.items():
    ms = sorted(v["mass"], reverse=True); ra.append(ms[0] / max(ms[1], 1e-3) if len(ms) > 1 else np.inf)
ra = np.array(ra); print(f"--- April 40 onset (A16 = A17, own predictions): chosen / second blob mass p10/25/50/75/90 {' / '.join(f'{x:.2f}' for x in np.quantile(ra[np.isfinite(ra)], [0.1, 0.25, 0.5, 0.75, 0.9]))}; single-blob windows {int((~np.isfinite(ra)).sum())}", flush=True)
