"""Local tuning of the same-month evidence rules on the NO-MEMORY NN (CAUSAL_MAP P7: without memory, the local
yardstick can see the value of evidence). Test-like conditions: observation layer 43.5% hidden, evidence from at most
3 earlier days of the month. Evaluates, on the selector-like train windows:
  onset  : T+30 probabilities of links without evidence x B, for evidence definitions (slot, dilation)
  ongoing: far predictions (> 3 links from a queue at T) kept only with evidence, for evidence definitions; and the
           near-region shrink for queues without same-month evidence (G7)
OOF: q25_nomem_s3+s4 (no-memory attention NN), or argv[1]."""
import sys, os, numpy as np, pandas as pd
sys.path.insert(0, r"C:/Users/user/tfb/src")
os.environ.setdefault("TFB_T2_FAR", "3")
import t2_eval_surprise as S, t2_eval_ongoing_shift as O
from t2_eval_oof import T, load_oof, far_mask
from cachelib import load

SPEC = sys.argv[1] if len(sys.argv) > 1 else "q25_nomem_s3+q25_nomem_s4"
oof = load_oof(SPEC); PC = {}
names = list(np.load("E:/tfb_nn/nn_data_afdem/names.npy", allow_pickle=True)); ci = names.index("last_queue")
X = np.load("E:/tfb_nn/nn_data_afdem/X.npy", mmap_mode="r"); OFF = np.load("E:/tfb_nn/nn_data_afdem/off.npy")


def qobs(p):
    if p not in PC:
        P = load(p, "train"); q = (P.V <= P.vcut[None, :]) & np.isfinite(P.V)
        PC[p] = q & (np.random.default_rng(0).random(q.shape) > 0.435)
    return PC[p]


DAYTYPE = os.environ.get("G8_DAYTYPE") == "1"                # evidence only from earlier days of the same day type


def evid(k, lo, hi, dil, sameday):
    p = str(T.META["panel"][k]); t = int(T.META["t"][k]); q = qobs(p); day = str(T.META["date"][k])[:10]
    seen = np.zeros(q.shape[1], bool); nd = 0
    if sameday:
        d0 = t - (t % 288) + 60
        if t > d0:
            seen |= q[d0:t].any(axis=0)
    for back in (1, 2, 3):
        dd = t - back * 288
        pday = np.datetime64(day) - np.timedelta64(back, "D")
        if DAYTYPE and ((pd.Timestamp(str(pday)).dayofweek >= 5) != (pd.Timestamp(day).dayofweek >= 5)):
            continue
        if dd + lo >= 0 and str(pday)[:7] == day[:7]:
            seen |= q[dd + lo:dd + hi].any(axis=0); nd += 1
    ev = seen.copy()
    for o in range(1, dil + 1):
        ev[o:] |= seen[:-o]; ev[:-o] |= seen[o:]
    return ev, nd


def bestset(p):
    pe = p.ravel(); o = np.argsort(-pe); cs = np.cumsum(pe[o]); kk = np.arange(1, pe.size + 1)
    kb = int(np.argmax(cs / (kk + pe.sum() - cs))) + 1; s = np.zeros(pe.size, bool); s[o[:kb]] = True
    return s.reshape(p.shape)


def iou(s, lab, el):
    u = ((s | lab) & el).sum()
    return 1.0 if u == 0 else (s & lab & el).sum() / u


# ---- onset
ON = {"none": None, "slot-30+60 dil2 (s45)": (-6, 12, 2), "slot-60+60 dil2": (-12, 12, 2), "slot-30+30 dil2": (-6, 6, 2),
      "slot-30+60 dil1": (-6, 12, 1), "slot-30+60 dil3": (-6, 12, 3), "slot-90+90 dil2": (-18, 18, 2)}
R = {n: [] for n in ON}
for k in S.SEL:
    if k not in oof:
        continue
    lab, el = T.LAB[k]; lab = lab.astype(bool); el = el.astype(bool); base = oof[k]
    for n, d in ON.items():
        pp = base.copy()
        if d is not None:
            ev, nd = evid(k, d[0], d[1], d[2], False)
            if nd:
                pp[-1] = pp[-1] * np.where(ev, 1.0, 0.5)
        R[n].append(iou(bestset(pp), lab, el))
b0 = np.mean(R["none"])
print(f"[onset n={len(R['none'])}] none {b0:.4f} | " + "  ".join(f"{n} {np.mean(v)-b0:+.4f}" for n, v in R.items() if n != "none"))

# ---- ongoing
OG = {"no rule": None, "blanket": "blanket", "gate s40 (day+slot-90+60 dil2)": (-18, 12, 2, True),
      "gate slot only": (-18, 12, 2, False), "gate day+slot-60+60": (-12, 12, 2, True), "gate dil1": (-18, 12, 1, True), "gate dil3": (-18, 12, 3, True)}
R = {n: [] for n in OG}; RS = []
for k in O.SEL:
    if k not in oof:
        continue
    lab, el = T.LAB[k]; lab = lab.astype(bool); el = el.astype(bool); s = bestset(oof[k]); far = far_mask(k)
    for n, d in OG.items():
        if d is None:
            ss = s
        elif d == "blanket":
            ss = s & ~far[None, :]
        else:
            ev, _ = evid(k, d[0], d[1], d[2], d[3]); ss = s & ~(far & ~ev)[None, :]
        R[n].append(iou(ss, lab, el))
    # near shrink for queues without evidence on earlier days (on top of the s40 gate)
    lq = np.asarray(X[OFF[k]:OFF[k + 1], ci], dtype=float) == 1
    ev, nd = evid(k, -18, 12, 0, False)
    if nd and lq.any() and not (ev & lq).any():
        near = np.zeros(len(lq), bool)
        for j in np.flatnonzero(lq):
            near[max(0, j - 3):j + 4] = True
        pp = oof[k].copy(); pp[:, near] *= 0.6
        evg, _ = evid(k, -18, 12, 2, True); s2 = bestset(pp) & ~(far & ~evg)[None, :]
        RS.append((iou(s & ~(far & ~evg)[None, :], lab, el), iou(s2, lab, el)))
b0 = np.mean(R["no rule"])
print(f"[ongoing n={len(R['no rule'])}] no rule {b0:.4f} | " + "  ".join(f"{n} {np.mean(v)-b0:+.4f}" for n, v in R.items() if n != "no rule"))
RS = np.array(RS)
if len(RS):
    print(f"[ongoing near shrink x0.6 on no-evidence queues, n={len(RS)}] gate {RS[:,0].mean():.4f} -> +shrink {RS[:,1].mean():.4f} ({RS[:,1].mean()-RS[:,0].mean():+.4f})")
