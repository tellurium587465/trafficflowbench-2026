"""G157 (11/01, ledger #117, follow-up of G154): the free-flow level of a link moves in March and April (sd 0.75-1.5 %, up to +-6 %) and it does not in the training
months (sd 0.02 %). (1) is it the free speed or the flow: x = r / r(train) against the month's free-cell flow / the training one, within panel. (2) labels: M3 dev
labels rebuilt with 0.6 x the link's March free-flow level instead of 0.6 x links.csv, cells that change. (3) the six pairs of V8 (t2_m3v2_gate) on M3 dev under both
labels. The hold-out days are not read; the meta file is not written.
usage: TFB_M3VER=v3 py -B -X utf8 g157_label_month.py"""
import os, sys
os.environ.setdefault("TFB_M3VER", "v3"); sys.dont_write_bytecode = True; sys.path.insert(0, r"C:/Users/user/tfb/src"); sys.argv = [sys.argv[0], "0,1,2,3"]
import numpy as np, pandas as pd
from cachelib import load, T2_PANELS
import t2_m3_gate as G, t2_m3v2_gate as V, m3v2

R = pd.read_csv(r"C:/Users/user/tfb/work/t2/g154_vf_month.csv"); rm = {p: g.sort_values("link").r.to_numpy() for p, g in R[R.month == "2031-03"].groupby("panel")}
tr = R[R.split == "train"].groupby(["panel", "link"]).r.median()
rows = []
for p in T2_PANELS:
    Pt = load(p, "train"); Qt = np.where(Pt.E & (Pt.V >= 0.85 * Pt.vf[None, :]), Pt.Q, np.nan); qt = np.nanmean(Qt, axis=0)
    for sp in ("validation", "private"):
        P = load(p, sp); Qm = np.where(P.E & (P.V >= 0.85 * P.vf[None, :]), P.Q, np.nan); qm = np.nanmean(Qm, axis=0); r = R[(R.panel == p) & (R.split == sp)].sort_values("link").r.to_numpy()
        ok = np.isfinite(r) & (qt > 0) & (qm > 0); x = np.log(r[ok] / tr[p].to_numpy()[ok]); f = np.log(qm[ok] / qt[ok])
        rows.append(dict(panel=p, split=sp, n=int(ok.sum()), corr=round(float(np.corrcoef(x, f)[0, 1]), 3), slope=round(float(np.polyfit(f, x, 1)[0]), 4), sd_flow=round(float(f.std()), 4), sd_x=round(float(x.std()), 4)))
    load.cache_clear()
print("x against the free-cell flow ratio"); print(pd.DataFrame(rows).to_string(index=False), flush=True)
DEV = [w for w in G.WID if not m3v2.is_holdout(np.array([G.META[w]["day"]]))[0]]; ch = np.zeros(4); bad = 0; wch = 0
for w in DEV:
    m = G.META[w]; P, _, _ = m3v2._panel(m["panel"]); t = m["t"]; y0 = m["y"].T.astype(bool); e = m["m"].T.astype(bool); Vh = P.V[t + 1:t + 7]
    y00 = e & (np.nan_to_num(Vh, nan=1e9) <= P.vcut[None, :]); bad += int((y00 != y0).sum())
    cut = P.vcut * np.where(np.isfinite(rm[m["panel"]]), rm[m["panel"]], 1.0); y1 = e & (np.nan_to_num(Vh, nan=1e9) <= cut[None, :]); m["yv"] = y1
    ch += [y0.sum(), (y1 & ~y0).sum(), (y0 & ~y1).sum(), e.sum()]; wch += int((y1 != y0).any())
print(f"labels: rebuilding with vcut reproduces y: {bad} cells differ | positives {ch[0]:.0f}, added {ch[1]:.0f} ({ch[1] / ch[0]:.4f}), removed {ch[2]:.0f} ({ch[2] / ch[0]:.4f}), of all labelled {(ch[1] + ch[2]) / ch[3]:.5f} | windows changed {wch} of {len(DEV)}", flush=True)
base_iou = G.iou


def iou_v(sel, m):
    y, mk = m["yv"], m["m"].T.astype(bool)
    if m["cond"] == "queue_onset":
        sel, y, mk = sel[-1:], y[-1:], mk[-1:]
    u = ((sel | y) & mk).sum(); return 1.0 if u == 0 else float((sel & y & mk).sum() / u)


def frame():
    cols = {k: {} for k in ("mem", "nomem", "ev_none", "ev_cur", "skip", "gate_off", "gate_on", "sim", "neg")}
    for w in DEV:
        m = G.META[w]
        if m["cond"] == "queue_onset":
            cols["mem"][w] = V.score_onset(G.MEM[w], m); cols["nomem"][w] = cols["ev_cur"][w] = V.score_onset(G.NOMEM[w], m)
            cols["ev_none"][w] = V.score_onset(G.NOMEM[w], m, "none"); cols["skip"][w] = V.score_onset(G.NOMEM[w], m, "skip"); cols["neg"][w] = V.score_onset(G.NEG[w], m)
        else:
            cols["mem"][w] = V.score_ongoing(G.MEM[w], m); cols["nomem"][w] = cols["gate_on"][w] = V.score_ongoing(G.NOMEM[w], m)
            cols["gate_off"][w] = V.score_ongoing(G.NOMEM[w], m, False); cols["sim"][w] = V.score_ongoing(G.SIM[w], m); cols["neg"][w] = V.score_ongoing(G.NEG[w], m)
    return V.frame(cols).dropna(subset=["nomem"])


D0 = frame(); G.iou = iou_v; Dv = frame(); G.iou = base_iou
ref = pd.read_csv(G.W + "m3v2_gatev3.csv").set_index("w"); d = max(float(np.abs(ref.loc[D0.w, c].to_numpy() - D0[c].to_numpy()).max()) for c in ("nomem", "gate_on"))
print(f"identity L0 against m3v2_gatev3: max |d| {d:.2e}", flush=True)
Dv.to_csv(G.W + "g157_pairs_labelv.csv", index=False); D0.to_csv(G.W + "g157_pairs_label0.csv", index=False)
on = D0.cond == "queue_onset"; og = ~on
for name, m_, a, b in (("1 memory -> no memory", D0.cond.notna(), "mem", "nomem"), ("2 evidence rule", on, "ev_none", "ev_cur"), ("3 far gate", og, "gate_off", "gate_on"), ("4 sim ongoing", og, "nomem", "sim"), ("5 B1 skip", on, "ev_cur", "skip"), ("6 negative control", D0.cond.notna(), "nomem", "neg")):
    s0 = m3v2.summary(D0[m_], a, b, "private"); sv = m3v2.summary(Dv[m_.to_numpy()], a, b, "private")
    print(f"{name:24s} L0 {s0['t2_weighted']:+.4f} (SE {s0['se']:.4f}) | Lv {sv['t2_weighted']:+.4f} (SE {sv['se']:.4f}) | sign kept {np.sign(s0['t2_weighted']) == np.sign(sv['t2_weighted'])}", flush=True)
