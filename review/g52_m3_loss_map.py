"""G52: M3 loss map (strategy session 10/22; the M3 version of G31, diagnostic only). Where does the March (real shift)
T2 score go? The production-like no-memory pipeline of the M3 gate: NOMEM blend (q24f, 5 configurations x seeds 0-3),
current evidence rule / far gate, k-day limit, trap 4; IoU on observed eligible cells (onset at T+30, ongoing 6 steps).
Exact loss decomposition: 1 - IoU = (FP + FN) / union, so every wrong cell carries 1 / union of its window's loss.
Cells (links of one time step, observed only) are put into runs of consecutive links ("components"):
  onset   FN in a true component that no predicted cell touches (+-1 link)  -> "missed / wrong place"
          FN in a true component that a predicted one touches                -> "extent: too small"
          FP in a predicted component that touches no true one              -> "wrong place / false alarm"
          FP in a predicted component that touches a true one               -> "extent: too large"
  ongoing the same, with "touches" read against the queue visible at T (last_queue, +-1 link):
          FN away from the visible queue -> "new detached queue missed"; FN next to it -> "growth under-predicted"
          FP next to the visible queue where the truth has nothing -> "queue that should vanish / shrink kept"
          FP away from it -> "spurious new queue"
Window classes (onset): both empty / missed (pred empty) / false alarm (truth empty) / wrong place (no overlap) /
extent (overlap, IoU < 1) / exact. Site rarity f for onset: the head (most downstream T+30 queued link // 3) in the same
panel x weekend x 2-hour slot among the TRAIN months' selector-like onset windows (t2_eval_surprise._cnt, all 9 months).
usage: python g52_m3_loss_map.py"""
import sys
sys.path.insert(0, r"C:/Users/user/tfb/src")
sys.argv = [sys.argv[0], "0,1,2,3"]
import numpy as np, pandas as pd
import t2_m3_gate as G
from m3_stats import weights, MARCH_FIRST_DOW
import t2_eval_surprise as S


def runs(v):
    """list of (start, end) inclusive runs of True in a 1-d bool array"""
    out, j = [], 0
    while j < len(v):
        if v[j]:
            e = j
            while e + 1 < len(v) and v[e + 1]:
                e += 1
            out.append((j, e)); j = e + 1
        else:
            j += 1
    return out


def touch(a, b, n):
    """cells of runs in a that touch (+-1 link) any cell of b"""
    bd = G.dil(b, 1); t = np.zeros(n, bool)
    for s, e in runs(a):
        if bd[s:e + 1].any():
            t[s:e + 1] = True
    return t


def f_site(m):
    y = m["y"].T[-1].astype(bool) & m["m"].T[-1].astype(bool)
    h = int(np.flatnonzero(y).max()) // 3 if y.any() else -1
    s = (m["panel"], ((MARCH_FIRST_DOW + m["day"]) % 7) >= 5, (m["t"] % 288) // 24)
    num = sum(c[h] for (ss, mm), c in S._cnt.items() if ss == s); den = sum(sum(c.values()) for (ss, mm), c in S._cnt.items() if ss == s)
    return (num + 0.5) / (den + 1.0)


rows = []
for w in G.WID:
    m = G.META[w]; c = m["cond"]; y = m["y"].T.astype(bool); mk = m["m"].T.astype(bool)
    if c == "queue_onset":
        sel = G.onset_pred(G.NOMEM[w], m, "cur")[-1:]; y, mk = y[-1:], mk[-1:]
    else:
        sel = G.ongoing_pred(G.NOMEM[w], m, True)
        i = G.POS[w]; lq = np.asarray(G.X[G.OFF[i]:G.OFF[i + 1], G.CI], dtype=float) == 1
    s, t_ = sel & mk, y & mk; u = (s | t_).sum(); iou = 1.0 if u == 0 else (s & t_).sum() / u
    cat = dict(fn_far=0, fn_near=0, fp_far=0, fp_near=0)
    for r in range(s.shape[0]):
        n = s.shape[1]; ref_fn = s[r] if c == "queue_onset" else lq; ref_fp = t_[r] if c == "queue_onset" else lq
        fn = t_[r] & ~s[r]; fp = s[r] & ~t_[r]
        near_fn = touch(t_[r], ref_fn, n); near_fp = touch(s[r], ref_fp, n)
        cat["fn_near"] += int((fn & near_fn).sum()); cat["fn_far"] += int((fn & ~near_fn).sum())
        cat["fp_near"] += int((fp & near_fp).sum()); cat["fp_far"] += int((fp & ~near_fp).sum())
    loss = {k: (v / u if u else 0.0) for k, v in cat.items()}
    assert abs(sum(loss.values()) - (1 - iou)) < 1e-9, (w, loss, iou)
    if c == "queue_onset":
        ps, ts = s[-1].any(), t_[-1].any()
        wc = ("both empty" if not ps and not ts else "missed (pred empty)" if not ps else "false alarm (truth empty)" if not ts
              else "wrong place (no overlap)" if not (s & t_).any() else "exact" if iou == 1 else "extent (overlap)")
    else:
        wc = "-"
    rows.append(dict(w=w, cond=c, panel=m["panel"], day=m["day"], k=m["k"], iou=iou, cls=wc,
                     f=f_site(m) if c == "queue_onset" and t_.any() else np.nan, **loss))
D = pd.DataFrame(rows); D["wt"] = weights(D); D["loss"] = 1 - D.iou
D.to_csv(G.W + "g52_m3_loss_map.csv", index=False)
NAMES = {"queue_onset": dict(fn_far="FN missed / wrong place", fn_near="FN extent too small", fp_far="FP wrong place / false alarm", fp_near="FP extent too large"),
         "queue_ongoing": dict(fn_far="FN new detached queue missed", fn_near="FN growth under-predicted", fp_far="FP spurious new queue", fp_near="FP queue that should vanish/shrink kept")}
for c, g in D.groupby("cond"):
    wl = (g.wt * g.loss).sum() / g.wt.sum()
    print(f"== {c}: windows {len(g)}, IoU raw {g.iou.mean():.4f} / weighted {1 - wl:.4f}; IoU=0 windows {np.mean(g.iou == 0):.3f} "
          f"carrying {g.loss[g.iou == 0].sum() / g.loss.sum():.2f} of the loss (weighted {(g.wt * g.loss)[g.iou == 0].sum() / (g.wt * g.loss).sum():.2f})")
    print("   loss by error type (raw / weighted share):", "; ".join(f"{NAMES[c][k]} {g[k].sum() / g.loss.sum():.2f} / {(g.wt * g[k]).sum() / (g.wt * g.loss).sum():.2f}" for k in NAMES[c]))
    if c == "queue_onset":
        t = g.groupby("cls").agg(n=("iou", "size"), iou=("iou", "mean"), loss=("loss", "sum")); t["share_windows"] = t.n / len(g); t["share_loss"] = t.loss / g.loss.sum()
        print(t[["n", "share_windows", "iou", "share_loss"]].round(3).to_string())
        gf = g[g.f.notna()]; rare = gf.f <= 0.1
        print(f"   rare sites (f <= 0.1, train months only; windows with a true queue {len(gf)}): share {rare.mean():.3f}, IoU {gf[rare].iou.mean():.4f} vs others {gf[~rare].iou.mean():.4f}, loss share {gf[rare].loss.sum() / g.loss.sum():.2f}")
    for name, key in (("k=0 (no earlier day) / k>=1", g.k == 0),):
        print(f"   {name}: windows {int(key.sum())} / {int((~key).sum())}, IoU {g[key].iou.mean():.4f} / {g[~key].iou.mean():.4f}, loss share {g[key].loss.sum() / g.loss.sum():.2f} / {g[~key].loss.sum() / g.loss.sum():.2f}")
    t = g.groupby("panel").agg(n=("iou", "size"), iou=("iou", "mean"), zero=("iou", lambda x: np.mean(x == 0)), loss=("loss", "sum"))
    t["loss_share"] = t.loss / g.loss.sum()
    for k in NAMES[c]:
        t[k] = g.groupby("panel")[k].sum() / g.groupby("panel").loss.sum()
    print(t.drop(columns="loss").round(3).to_string())
