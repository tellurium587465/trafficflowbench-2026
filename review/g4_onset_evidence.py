"""Gem check G4 (knowledge-map gap: T2 onset x same-month pre-T evidence): onset = the first queue of the day, so the
same day carries no evidence, but up to 3 earlier days of the same month do. Links (+-2) that queued in the same slot
(T-30 .. T+60 min) on those days get their T+30 probability multiplied by a; the others by b. Train onset windows
(selector-like), s31-like blend OOF; all windows and rare-site windows (f < 0.3), weekday / weekend."""
import sys, os, numpy as np, pandas as pd
sys.path.insert(0, r"C:/Users/user/tfb/src")
import t2_eval_surprise as S
from t2_eval_oof import T, load_oof
from cachelib import load

SPEC = sys.argv[1] if len(sys.argv) > 1 else "l127_full_af:2+q14_attn1_glob+q16_attnglob_s1+q14_attn1_wp05:2+q13_attn1:2+q12_af_r13_nopos+q17_drop1"
oof = load_oof(SPEC); PC = {}


def iou_of(p, lab, el):
    pe = p.ravel(); o = np.argsort(-pe); cs = np.cumsum(pe[o]); kk = np.arange(1, pe.size + 1)
    kb = int(np.argmax(cs / (kk + pe.sum() - cs))) + 1; s = np.zeros(pe.size, bool); s[o[:kb]] = True; s = s.reshape(p.shape)
    u = ((s | lab) & el).sum()
    return 1.0 if u == 0 else (s & lab & el).sum() / u


GRID = [(1.0, 1.0), (1.0, 0.7), (1.0, 0.5), (1.0, 0.35), (1.0, 0.2), (1.3, 0.5)]
rows = []
for j, k in enumerate(S.SEL):
    if k not in oof:
        continue
    p = str(T.META["panel"][k]); t = int(T.META["t"][k])
    if p not in PC:
        P = load(p, "train"); PC[p] = (P.V <= P.vcut[None, :]) & np.isfinite(P.V)
        if os.environ.get("G4_MASK"):                              # observation layer as sparse as in the test months
            PC[p] = PC[p] & (np.random.default_rng(0).random(PC[p].shape) > float(os.environ["G4_MASK"]))
    q = PC[p]; day = str(T.META["date"][k])[:10]; seen = np.zeros(q.shape[1], bool); nd = 0
    for back in (1, 2, 3):
        dd = t - back * 288
        if dd - 6 >= 0 and str(np.datetime64(day) - np.timedelta64(back, "D"))[:7] == day[:7]:
            seen |= q[dd - 6:dd + 12].any(axis=0); nd += 1
    ev = seen.copy()
    for o in (1, 2):
        ev[o:] |= seen[:-o]; ev[:-o] |= seen[o:]
    lab, el = T.LAB[k]; lab = lab.astype(bool); el = el.astype(bool); base = oof[k]
    r = []
    for a, b in GRID:
        pp = base.copy(); pp[-1] = np.clip(pp[-1] * np.where(ev, a, b), 0, 1)
        r.append(iou_of(pp, lab, el))
    rows.append((S.F[j] < 0.3, pd.Timestamp(day).dayofweek >= 5, nd > 0, *r))
R = pd.DataFrame(rows, columns=["rare", "weekend", "hasprior"] + [f"a{a}_b{b}" for a, b in GRID])
cols = R.columns[3:]
for lab_, m in (("all", R.index == R.index), ("has prior days", R.hasprior), ("rare f<0.3", R.rare), ("weekday", ~R.weekend), ("weekend", R.weekend)):
    r = R[m]; base = r[cols[0]].mean()
    print(f"{lab_:15s} n={len(r):4d} base {base:.4f} | " + "  ".join(f"{c} {r[c].mean() - base:+.4f}" for c in cols[1:]))
