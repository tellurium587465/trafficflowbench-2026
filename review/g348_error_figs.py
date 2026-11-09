"""G348 (11/05, ledger #275): error-analysis figures, one PNG per window, from G347's dumps (the k draw of the largest weight).
Panels (links in travel order, upstream left; the official history is T-60 .. T-5, row T is neither released nor scored): 1 speed T-60 .. T+30 (history and horizon are the true speeds; grey = not finite / the
Task 1 blank in the training months' history), on-ramps (^) / off-ramps (v) marked; 2 the true queue 0/1 (queue_matrix) T-60 .. T+30;
3 the horizon probabilities the decode sees; 4 the decoded set against the truth (TP green, FP red, FN blue, unscored grey).
Onset: the true queue of the same clock band on the 3 previous days as small strips. Title: window id, condition, day, origin time,
IoU (k-expected, and of the drawn k); x cropped to the cells with truth / p > 0.05 / the set / the box +-20 links, persistence IoU (truth at T-1 held), per-row IoU, box length / age (ongoing), evidence acts (onset).
Windows: the 80 official June windows; M3 dev onset misses (< 0.5) and ongoing worst 10; January / February onset misses and ongoing worst
15; 5 hits (IoU >= 0.95, seed 0) per instrument x condition for M3 and January / February.
-> Desktop/trafficflowbench/erroranalysis_20261105/<instrument>/*.png and index.csv. usage: py -B -X utf8 g348_error_figs.py"""
import os, sys, pickle
os.environ["TFB_T2CACHE"] = "2_noT_af"; os.environ.setdefault("TFB_M3VER", "v3")
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
sys.path.insert(0, r"C:/Users/user/tfb/src")
from cachelib import load
from t2_windows import queue_matrix
plt.rcParams["font.family"] = ["Yu Gothic", "MS Gothic", "DejaVu Sans"]
W = "C:/Users/user/tfb/work/t2/"; OUT = "C:/Users/user/OneDrive/Desktop/trafficflowbench/erroranalysis_20261105/"
A = pickle.load(open(W + "g342_jun80.pkl", "rb")); WID = {r["k"]: r["wid"] for r in A["rows"]}; NWIN = A["NWIN"]
ON_MJ = pickle.load(open(W + "g347_on_m3jf.pkl", "rb")); ON_J = pickle.load(open(W + "g347_on_jun80.pkl", "rb"))
OG_MJ = pickle.load(open(W + "g347_og_m3jf.pkl", "rb")); OG_J = pickle.load(open(W + "g347_og_jun80.pkl", "rb"))
KE = {}
for f, c in (("g347_on_m3jf_m.csv", "base"), ("g347_on_m3jf_t.csv", "L_base"), ("g347_on_jun80_t.csv", "L_base"), ("g347_og_m3jf_M3_dev_ongoing.csv", "base"),
             ("g347_og_m3jf_January_February_ongoing.csv", "base"), ("g347_og_jun80_June_2030_ongoing.csv", "base")):
    d = pd.read_csv(W + f)
    for w, v in zip(d.w, d[c]):
        KE[str(w)] = float(v)
rng = np.random.default_rng(0); sel = []
mo = {str(k): v for k, v in ON_MJ.items() if isinstance(k, str)}; jo = {str(k): v for k, v in ON_MJ.items() if not isinstance(k, str)}
mg = {str(k): v for k, v in OG_MJ.items() if isinstance(k, str)}; jg = {str(k): v for k, v in OG_MJ.items() if not isinstance(k, str)}
for k, v in ON_J.items():
    sel.append(("JUN", "queue_onset", str(k), v, "公式"))
for k, v in OG_J.items():
    if not isinstance(k, str) and k >= NWIN:
        sel.append(("JUN", "queue_ongoing", str(k), v, "公式"))
for inst, cond, D_, how, n_ in (("M3", "queue_onset", mo, "miss", 0), ("M3", "queue_ongoing", mg, "worst", 10), ("JF", "queue_onset", jo, "miss", 0), ("JF", "queue_ongoing", jg, "worst", 15)):
    ks = sorted(D_, key=lambda x: KE[x]); bad = [x for x in ks if KE[x] < 0.5] if how == "miss" else ks[:n_]
    hits = [x for x in ks if KE[x] >= 0.95]; hits = list(rng.choice(hits, min(5, len(hits)), replace=False))
    sel += [(inst, cond, x, D_[x], "外れ" if how == "miss" else "最悪") for x in bad] + [(inst, cond, x, D_[x], "当たり") for x in hits]
print("figures:", len(sel), pd.Series([s[0] + " " + s[1] + " " + s[4] for s in sel]).value_counts().to_dict(), flush=True)
iou = lambda s, y, e: 1.0 if ((s | y) & e).sum() == 0 else float((s & y & e).sum() / ((s | y) & e).sum())
CM4 = ListedColormap(["white", "#d0d0d0", "#2ca02c", "#d62728", "#1f77b4"])
PC = {}


def panel(p, split):
    if (p, split) not in PC:
        if len(PC) > 3:
            PC.clear(); load.cache_clear()
        P = load(p, split); PC[(p, split)] = (P, queue_matrix(P).astype(bool))
    return PC[(p, split)]


rows = []; os.makedirs(OUT, exist_ok=True)
for inst, cond, key, v, tag in sel:
    if cond == "queue_onset":
        pr, kk, q, st, chg, y, e, p, t, day, split = v; lq = None
    else:
        pr, kk, q, st, lq, y, e, p, t, day, split = v; chg = np.nan
    P, Qm = panel(p, split); n = q.shape[1]; y = y.astype(bool); e = e.astype(bool); st = np.asarray(st).astype(bool); assert n == P.n, (n, P.n)
    r0, r1 = t - 12, t + 7; Vw = P.V[r0:r1].astype(float).copy()
    if split == "train":
        Vw[:13][P.blank[r0:t + 1]] = np.nan
    Qw = Qm[r0:r1].copy(); Qw[13:] = y; ps = Qm[t - 1]
    rio = [iou(st[r], y[r], e[r]) for r in range(6)]; pio = [iou(ps, y[r], e[r]) for r in range(6)]
    wi, wp = iou(st, y, e), iou(np.broadcast_to(ps, y.shape), y, e)
    if cond == "queue_onset":
        wi, wp = iou(st[-1], y[-1], e[-1]), iou(ps, y[-1], e[-1])
    if lq is not None:
        L = int(lq.sum()); age = 0
        for r in range(t - 1, max(t - 289, 0), -1):
            if (Qm[r] & lq).any():
                age += 1
            else:
                break
    else:
        L, age = 0, 0
    wid = WID.get(int(key), key) if inst == "JUN" else key; tm = P.times[t]; dom = int(tm[8:10])
    ttl = (f"{inst} {wid} | {cond} | {tm[:10]} (月の {dom} 日目) {tm[11:16]} | {tag} | IoU k期待 {KE.get(key, np.nan):.3f} / 引いた k={kk} (重み {pr:.2f}) {wi:.3f} | 持続 {wp:.3f}\n"
           f"行ごと IoU T+5..T+30: " + " ".join(f"{x:.2f}" for x in rio) + " | 持続: " + " ".join(f"{x:.2f}" for x in pio)
           + (f" | 箱 長さ {L} 年齢 {age} 行" if lq is not None else f" | 根拠が働く {bool(chg > 0.5)}"))
    nstrip = 3 if cond == "queue_onset" else 0
    fig = plt.figure(figsize=(16, 9 + 1.2 * nstrip)); gs = fig.add_gridspec(4 + nstrip, 1, height_ratios=[19, 19, 7, 7] + [6] * nstrip, hspace=0.45)
    a1 = fig.add_subplot(gs[0]); im = a1.imshow(Vw, aspect="auto", cmap="RdYlGn", vmin=0, vmax=110, interpolation="nearest", extent=(-0.5, n - 0.5, 6.5, -12.5))
    a1.axhline(-0.5, color="k", lw=1.2); a1.axhline(0.5, color="k", lw=1.0, ls="--"); a1.set_ylabel("行 (T からの 5 分)"); a1.set_title("1 速度 km/h(灰 = 空白;実線 = 見える履歴の終わり T−5、破線 = 地平の始まり T+5、行 0 = T は非公開・採点外)", fontsize=9, loc="left")
    for j in np.flatnonzero(P.on_attached):
        a1.plot(j, -12.9, marker="^", color="purple", ms=4, clip_on=True)
    for j in np.flatnonzero(P.off_attached):
        a1.plot(j, -12.9, marker="v", color="orange", ms=4, clip_on=True)
    if lq is not None:
        for j in np.flatnonzero(lq):
            a1.plot(j, 6.3, marker="s", color="k", ms=3, clip_on=True)
    fig.colorbar(im, ax=a1, fraction=0.015, pad=0.005)
    a2 = fig.add_subplot(gs[1], sharex=a1); a2.imshow(Qw, aspect="auto", cmap="Greys", vmin=0, vmax=1, interpolation="nearest", extent=(-0.5, n - 0.5, 6.5, -12.5)); a2.axhline(-0.5, color="r", lw=1.2); a2.axhline(0.5, color="r", lw=1.0, ls="--")
    a2.set_title("2 真の渋滞 0/1(赤の実線 = 見える履歴の終わり、破線 = 地平の始まり)", fontsize=9, loc="left")
    a3 = fig.add_subplot(gs[2], sharex=a1); im3 = a3.imshow(q, aspect="auto", cmap="magma_r", vmin=0, vmax=1, interpolation="nearest", extent=(-0.5, n - 0.5, 6.5, 0.5)); fig.colorbar(im3, ax=a3, fraction=0.015, pad=0.005)
    a3.set_title("3 地平の予測確率(復号が見る値)", fontsize=9, loc="left")
    cat = np.where(~e, 1, np.where(st & y, 2, np.where(st & ~y, 3, np.where(~st & y, 4, 0))))
    a4 = fig.add_subplot(gs[3], sharex=a1); a4.imshow(cat, aspect="auto", cmap=CM4, vmin=0, vmax=4, interpolation="nearest", extent=(-0.5, n - 0.5, 6.5, 0.5))
    a4.set_title("4 復号の集合と正解(緑 TP・赤 FP・青 FN・灰 採点外)", fontsize=9, loc="left"); a4.set_xlabel("リンク(走行方向、上流 → 下流)")
    for i in range(nstrip):
        d0 = t - 288 * (i + 1)
        if d0 - 12 >= 0:
            a = fig.add_subplot(gs[4 + i], sharex=a1); a.imshow(Qm[d0 - 12:d0 + 7], aspect="auto", cmap="Blues", vmin=0, vmax=1, interpolation="nearest", extent=(-0.5, n - 0.5, 6.5, -12.5)); a.axhline(-0.5, color="r", lw=0.8)
            a.set_title(f"{i + 1} 日前の同じ時刻帯の真の渋滞", fontsize=8, loc="left")
    rel = Qw.any(axis=0) | (q > 0.05).any(axis=0) | st.any(axis=0) | (lq if lq is not None else False)
    if rel.any():
        x0, x1 = max(int(np.flatnonzero(rel).min()) - 20, 0), min(int(np.flatnonzero(rel).max()) + 20, n - 1); a1.set_xlim(x0 - 0.5, x1 + 0.5)
    ttl += f" | 表示リンク {a1.get_xlim()[0] + 0.5:.0f}..{a1.get_xlim()[1] - 0.5:.0f} / {n}"
    fig.suptitle(ttl, fontsize=9, x=0.01, ha="left"); d_ = OUT + inst + "/"; os.makedirs(d_, exist_ok=True)
    fn = f"{inst}_{'on' if cond == 'queue_onset' else 'og'}_{tag}_{wid}.png"; fig.savefig(d_ + fn, dpi=80, bbox_inches="tight"); plt.close(fig)
    ys = (lq is not None) and (L <= 3 or age <= 3)
    rows.append(dict(instrument=inst, window=wid, key=key, panel=p, condition=cond, tag=tag, date=tm[:10], day_of_month=dom, origin=tm[11:16], iou_kexp=KE.get(key, np.nan), k_drawn=kk, k_weight=pr,
                     iou_drawn=wi, iou_persist=wp, **{f"iou_r{r + 1}": rio[r] for r in range(6)}, **{f"persist_r{r + 1}": pio[r] for r in range(6)}, **{f"diff_r{r + 1}": rio[r] - pio[r] for r in range(6)},
                     box_len=L, box_age=age, young_small=ys, prior_days=dom - 1, evidence_acts=(bool(chg > 0.5) if cond == "queue_onset" else ""), true_cells=int((y & e).sum()), pred_cells=int((st & e).sum()), file=inst + "/" + fn))
D = pd.DataFrame(rows); D.to_csv(OUT + "index.csv", index=False, encoding="utf-8-sig"); print("written", len(D), "figures ->", OUT, flush=True)
