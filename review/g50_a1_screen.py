"""G50: A1 screen (A1_PREREG_20261022.md). Month-CV fold 2 OOF of single NN runs (FOLDS=2), scored on the real
selector-like windows of Dec-Feb: ongoing IoU over the six steps, onset IoU at T+30 (as the production decoder), with
the expected-IoU set. Runs are grouped by name prefix (e.g. a1_real_b3 -> seeds averaged as probabilities before
decoding when the group has several seeds). Differences against the baseline group, cluster bootstrap (panel x day).
The synthetic windows are training-only (fold_month puts them in their template month; they are never scored).
usage: python g50_a1_screen.py <baseline group> <group> [group ...]   (group = tag prefix before _s<seed>)"""
import sys, glob, os, numpy as np, pandas as pd
sys.path.insert(0, r"C:/Users/user/tfb/src")
from t2_eval_oof import T, oof_of
from t2_model import best_set

W = "C:/Users/user/tfb/work/t2/"
FOLD2 = ("2030-12", "2031-01", "2031-02")
groups = sys.argv[1:]


def members(g):
    tags = sorted(os.path.basename(f)[4:-4] for f in glob.glob(W + f"oof_{g}_s[0-9].pkl"))
    if not tags and os.path.exists(W + f"oof_{g}.pkl"):
        tags = [g]
    return tags


def iou(p, y, e, cond):
    q = p.copy()
    if cond == "queue_onset":
        q[:5] = 0
    s = best_set(q.ravel(), np.ones(q.size, bool)).reshape(q.shape)
    u = ((s | y) & e).sum(); return 1.0 if u == 0 else (s & y & e).sum() / u


O = {g: [oof_of(t) for t in members(g)] for g in groups}
for g, os_ in O.items():
    print(g, "members", members(g), flush=True)
    assert os_, f"no OOF for {g}"
keys = sorted(set.intersection(*[set(o) for os_ in O.values() for o in os_]))
rows = []
for k in keys:
    if not T.META["is_sel"][k] or str(T.META["date"][k])[:7] not in FOLD2:
        continue
    cond = str(T.META["cond"][k]); t = int(T.META["t"][k])
    if cond == "queue_onset" and (t + 6) % 288 >= 144:
        continue
    lab, el = T.LAB[k]; y = lab.astype(bool); e = el.astype(bool)
    row = dict(cond=cond, cl=str(T.META["panel"][k]) + str(T.META["date"][k])[:10])
    for g, os_ in O.items():
        row[g] = iou(np.mean([o[k] for o in os_], axis=0), y, e, cond)
    rows.append(row)
D = pd.DataFrame(rows); rng = np.random.default_rng(0); base = groups[0]
for cond, gd in D.groupby("cond"):
    u = gd.cl.unique(); idx = {x: np.flatnonzero(gd.cl.to_numpy() == x) for x in u}
    print(f"{cond}: windows {len(gd)}, " + ", ".join(f"{g} {gd[g].mean():.4f}" for g in groups))
    for g in groups[1:]:
        d = (gd[g] - gd[base]).to_numpy()
        se = np.std([d[np.concatenate([idx[x] for x in rng.choice(u, len(u))])].mean() for _ in range(1000)])
        print(f"   {g} - {base}: {d.mean():+.4f} (SE {se:.4f}, {d.mean() / max(se, 1e-9):+.1f} SE)", flush=True)
