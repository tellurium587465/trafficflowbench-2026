"""G244 (11/03, ledger #207, the strategy's (1); ruling 748784: "The official label is computed from the underlying state rather than
the released observation, so observation noise near the threshold will cause small differences"). Our labels are observed speed
<= 0.6 v_f. A less noisy speed from the same cells: v_dn = mean of the observed speed and the occupancy speed Q / (285 occ) (total
flow; its median ratio to the observed speed is 1.00 in every band near the threshold on four corridors), only where the observed
speed is finite (the same support as the labels). Instrument = G229's (#197): the 316 January / February selector ongoing windows,
capB0 s0-3 on the unmasked-history inputs, G119's SCG flow with the row T in the post-processing (#197, as A11 / A12).
  (1) flips between the two labels on eligible finite horizon cells: rate, direction, share near the threshold, share at a queue edge
  (2) IoU of the same decoded sets against both labels
  (3) the decode knobs on both labels: temperature 1.0 / 1.2 / 1.4 and a scale 0.85 / 1.0 / 1.15 on the probabilities right before
      best_set (G119's decode patched), each cell against the production cell (1.2, 1.0) under each label
Training split only (no test month, no hold-out). Identity: the rebuilt observed label == T.LAB, cell (1.0, 1.0) == G229's trow.
usage: TFB_M3VER=v3 py -B -X utf8 g244_label_denoise.py"""
import os, sys, pickle, importlib.util
os.environ.setdefault("TFB_M3VER", "v3")
import numpy as np, pandas as pd
R = r"C:/Users/user/tfb/"


def module(nm, path, argv):
    a = sys.argv; sys.argv = argv; spec = importlib.util.spec_from_file_location(nm, path); mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); sys.argv = a
    return mod


g = module("g119", R + "review/g119_month_postproc.py", ["g119", "run"]); T, MON = g.T, g.MON
KS = [k for k in np.flatnonzero(T.META["is_sel"] & np.isin(MON, g.MONTHS)) if str(T.META["cond"][k]) == "queue_ongoing"]
PR = [pickle.load(open(g.W + f"probs_jfmh0__capB0_s{s}_fold2030-12.pkl", "rb")) for s in range(4)]
lg = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6))); sg = lambda x: 1 / (1 + np.exp(-x))
dec0 = g.decode; KN = [1.0, 1.0]


def dec(pb):
    te, al = KN; q = pb if te == 1.0 else sg(lg(pb) / te); return dec0(np.clip(q * al, 0.0, 1 - 1e-6) if al != 1.0 else q)


g.decode = dec
GRID = [(te, al) for te in (1.0, 1.2, 1.4) for al in (0.85, 1.0, 1.15)]; rows = []; FL = dict(cells=0, flip=0, f10=0, f01=0, near=0, edge=0, qo=0, qd=0); bad = 0
for mo in g.MONTHS:
    for p in sorted({str(T.META["panel"][k]) for k in KS if MON[k] == mo}):
        P = g.load(p, "train"); key = (mo, p, "mask"); g.CTX.clear(); g.sim_blank.cache_clear(); g.CTX[key] = g.Ctx(key, P, np.where(P.blank, np.nan, P.V), P.V); c = g.CTX[key]
        with np.errstate(all="ignore"):
            vo = P.Q / (285.0 * P.O)
        vdn = np.where(np.isfinite(vo) & (P.O > 0), (P.V + vo) / 2, P.V)
        for k in [k for k in KS if MON[k] == mo and str(T.META["panel"][k]) == p]:
            t = int(T.META["t"][k]); dom = int(P.times[t][8:10]) - 1; lab, el = T.LAB[k]; y, e = lab.astype(bool), el.astype(bool)
            yo = (np.nan_to_num(P.V[t + 1:t + 7], nan=1e9) <= P.vcut[None, :]) & e; bad += int((yo != y).sum())
            yd = (np.nan_to_num(vdn[t + 1:t + 7], nan=1e9) <= P.vcut[None, :]) & e
            fin = e & np.isfinite(P.V[t + 1:t + 7]); fl = (y != yd) & fin; x = P.V[t + 1:t + 7] / P.vcut[None, :]
            ed = np.zeros_like(y); ed[:, 1:] |= y[:, 1:] != y[:, :-1]; ed[:, :-1] |= y[:, :-1] != y[:, 1:]
            FL["cells"] += int(fin.sum()); FL["flip"] += int(fl.sum()); FL["f10"] += int((fl & y).sum()); FL["f01"] += int((fl & ~y).sum())
            FL["near"] += int((fl & (np.abs(np.nan_to_num(x, nan=9) - 1) < 0.2)).sum()); FL["edge"] += int((fl & ed).sum()); FL["qo"] += int((y & fin).sum()); FL["qd"] += int((yd & fin).sum())
            lq = np.asarray(g.XN[g.OFFN[k]:g.OFFN[k + 1]][:, g.G.CI], float) == 1; ot = c.obs[t].astype(bool); lq2 = np.where(ot, c.q[t].astype(bool), lq)
            p_ = sum(d[str(k)] for d in PR) / len(PR); cone = g.cone_parts(c, t, lq2, P.length); r = dict(w=int(k), month=mo, panel=p, day=dom, flips=int(fl.sum()))
            for te, al in GRID:
                KN[:] = [te, al]; so = sd = 0.0
                for kk, pr in g.m3v2.kdist(dom, cap=3).items():
                    sel = g.ongoing_pred(c, p_ * g.cone_f(c, cone, t, kk), t, kk, True, lq2); so += pr * g.iou(sel, y, e, "queue_ongoing"); sd += pr * g.iou(sel, yd, e, "queue_ongoing")
                r[f"o_{te}_{al}"] = so; r[f"d_{te}_{al}"] = sd
            rows.append(r)
        g.load.cache_clear()
KN[:] = [1.0, 1.0]; D = pd.DataFrame(rows); D.to_csv(g.W + "g244_jf.csv", index=False)
ref = pd.read_csv(g.W + "g229_jf.csv").set_index("w").trow.reindex(D.w).to_numpy()
print(f"{len(D)} windows | identity: rebuilt observed label vs T.LAB cells differing {bad} | cell (1.0, 1.0) vs G229 trow max |d| {np.abs(ref - D['o_1.0_1.0'].to_numpy()).max():.2e}", flush=True)


def boot(d, cl, n_boot=2000, seed=0):
    rng = np.random.default_rng(seed); ids = pd.factorize(cl)[0]; M = np.bincount(ids, d); N = np.bincount(ids)
    Rm = rng.multinomial(len(M), np.full(len(M), 1.0 / len(M)), size=n_boot); return float(((Rm @ M) / (Rm @ N)).std())


cl = (D.month.astype(str) + "_" + D.panel + "_" + D.day.astype(str)).to_numpy()
print(f"(1) eligible finite horizon cells {FL['cells']}: flips {FL['flip']} ({FL['flip'] / FL['cells']:.4%}), queue -> free {FL['f10']}, free -> queue {FL['f01']} | "
      f"of the flips near the threshold (|v / v_cut - 1| < 0.2) {FL['near'] / max(FL['flip'], 1):.1%}, at a queue edge {FL['edge'] / max(FL['flip'], 1):.1%} | queue cells observed {FL['qo']} denoised {FL['qd']} | windows with a flip {int((D.flips > 0).sum())}")
for cell in ("1.0_1.0", "1.2_1.0"):
    d = (D[f"d_{cell}"] - D[f"o_{cell}"]).to_numpy(); print(f"(2) cell {cell}: IoU observed label {D[f'o_{cell}'].mean():.4f}, denoised label {D[f'd_{cell}'].mean():.4f}, paired {d.mean():+.4f} (SE {boot(d, cl):.4f})")
print("(3) each cell minus the production cell (1.2, 1.0), observed label | denoised label")
for te, al in GRID:
    c_ = f"{te}_{al}"; do = (D[f"o_{c_}"] - D["o_1.2_1.0"]).to_numpy(); dd = (D[f"d_{c_}"] - D["d_1.2_1.0"]).to_numpy()
    print(f"  T {te} scale {al}: {do.mean():+.4f} (SE {boot(do, cl):.4f}) | {dd.mean():+.4f} (SE {boot(dd, cl):.4f})")
bo = max(GRID, key=lambda x: D[f"o_{x[0]}_{x[1]}"].mean()); bd = max(GRID, key=lambda x: D[f"d_{x[0]}_{x[1]}"].mean()); print("best cell: observed label", bo, "| denoised label", bd)
