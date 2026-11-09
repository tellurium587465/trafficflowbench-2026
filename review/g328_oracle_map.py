"""G328 (11/05, ledger #262, measurement only): the T2 headroom map by oracle decomposition, on three instruments (M3 v3 dev = A17's mix;
January / February = capB0; June 2030 official windows = cvf fold 2030-06), the same decode and gate as each instrument's base, k expectation.
ongoing (temperature 0.8 -> best_set -> far gate): the cells split disjointly by the last-queue runs lq (links upstream -> downstream; a run's
tail = its upstream end, head = downstream end): S_box = the links of the runs, S_tail = up to 3 links upstream of each tail (outside the runs),
S_new = the rest. O_box / O_tail / O_new: that set's cells set to the truth (0/1); O_box+tail. O_map / O_map_any on S_new: S_new cells off the
month's other-day heads (+-1 link; within +-1 h of the window's time / at any time) set to 0.
onset (the T+30 row just before best_set): O_loc = 0 outside +-2 links of the true cells; O_size = the true count k given, the top k among the scored cells taken;
O_loc+size; O_map / O_map_any = 0 off the month's other-day heads (+-1 link; +-1 h / any time).
Per oracle: mean IoU change, windows changed, share of (1 - base IoU). Not used to accept anything. usage: TFB_M3VER=v3 py -B -X utf8 g328_oracle_map.py"""
import os, sys, pickle, importlib.util
os.environ.setdefault("TFB_M3VER", "v3")
import numpy as np, pandas as pd
R0 = r"C:/Users/user/tfb/"; sys.path.insert(0, R0 + "src")
import t2_m3_gate as G, m3v2
from t2_model import best_set
W = R0 + "work/t2/"
lg = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6))); tmp = lambda p, T: 1 / (1 + np.exp(-lg(p) / T))
bs = lambda pb: best_set(pb.ravel(), np.ones(pb.size, bool)).reshape(pb.shape)
dil = lambda m, r: np.convolve(m.astype(int), np.ones(2 * r + 1, int), mode="same") > 0


def module(nm, path, argv):
    a = sys.argv; sys.argv = argv; spec = importlib.util.spec_from_file_location(nm, path); mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); sys.argv = a
    return mod


def head_rows(V, vcut):
    fin = np.isfinite(V); q = fin & (V <= vcut[None, :]); return q & ~np.concatenate([q[:, 1:], np.zeros((len(q), 1), bool)], axis=1)


def maps(H, day, tod):
    """H (rows of the month, links) heads, day index and time-of-day row of the window -> (+-1 h set, any-time set), each dilated +-1 link"""
    nd = H.shape[0] // 288; other = [d for d in range(nd) if d != day]
    if not other:
        z = np.zeros(H.shape[1], bool); return z, z
    Hd = H[:nd * 288].reshape(nd, 288, -1)[other]; lo, hi = max(tod - 12, 0), min(tod + 13, 288)
    return dil(Hd[:, lo:hi].any(axis=(0, 1)), 1), dil(Hd.any(axis=(0, 1)), 1)


def og_sets(lq):
    n = len(lq); box = lq.copy(); tail = np.zeros(n, bool)
    for j in [j for j in range(n) if lq[j] and (j == 0 or not lq[j - 1])]:
        tail[max(j - 3, 0):j] = True
    tail &= ~box; return box, tail, ~box & ~tail


OG = ("O_box", "O_tail", "O_new", "O_box+tail", "O_map", "O_map_any"); ON = ("O_loc", "O_size", "O_loc+size", "O_map", "O_map_any")


def og_variants(p, y, sets, mp, mpa):
    box, tail, new = sets; yf = y.astype(float); out = {}
    for nm, m in (("O_box", box), ("O_tail", tail), ("O_new", new), ("O_box+tail", box | tail)):
        q = p.copy(); q[:, m] = yf[:, m]; out[nm] = q
    q = p.copy(); q[:, new & ~mp] = 0.0; out["O_map"] = q; q = p.copy(); q[:, new & ~mpa] = 0.0; out["O_map_any"] = q
    return out


def on_sel(row, y, e, mp, mpa):
    yt = y & e; k = int(yt.sum()); dec = lambda r: best_set(r, np.ones(r.size, bool))
    loc = dil(yt, 2); rl = np.where(loc, row, 0.0); top = lambda r: (np.argsort(-np.where(e, r, -1.0))[:k] if k else np.zeros(0, int))   # top k among the scored cells
    s_size = np.zeros(row.size, bool); s_size[top(row)] = True; s_ls = np.zeros(row.size, bool); s_ls[top(rl)] = True
    return dict(base=dec(row), O_loc=dec(rl), O_size=s_size, O_loc_size=s_ls, O_map=dec(np.where(mp, row, 0.0)), O_map_any=dec(np.where(mpa, row, 0.0)))


iou1 = lambda s_, y, e: 1.0 if ((s_ | y) & e).sum() == 0 else float((s_ & y & e).sum() / ((s_ | y) & e).sum())


def table(rows, names, lab):
    D = pd.DataFrame(rows); b = D.base; out = [f"--- {lab}: {len(D)} windows, base IoU {b.mean():.4f} (headroom 1 - base {1 - b.mean():.4f})"]
    for nm in names:
        d = D[nm] - b; out.append(f"  {nm:11s}: {D[nm].mean():.4f} ({d.mean():+.4f}; windows changed {int((np.abs(d) > 1e-12).sum())}; share of the headroom {d.mean() / max(1 - b.mean(), 1e-9):+.1%})")
    print("\n".join(out), flush=True); return D


g132 = module("g132", R0 + "review/g132_p3_grid.py", ["g132", "run"]); h = module("g119", R0 + "review/g119_month_postproc.py", ["g119", "run"]); T_ = h.T; MON = h.MON
# --- M3 dev (A17): month heads from the released March data
from cachelib import load as cload, T2_PANELS
HM = {}
for p in T2_PANELS:
    P = cload(p, "validation"); HM[p] = head_rows(P.V, P.vcut); cload.cache_clear()
D4 = pickle.load(open(W + "g284_og_pretemp_dev.pkl", "rb")); rows = []
for w, v in D4.items():
    m = G.META[w]; i_ = G.POS[w]; lq = np.asarray(G.X[G.OFF[i_]:G.OFF[i_ + 1], G.CI], float) == 1; sets = og_sets(lq); mp, mpa = maps(HM[m["panel"]], m["day"], m["t"] % 288)
    y = (m["y"].T.astype(bool) & m["m"].T.astype(bool)); r = dict(w=w, base=0.0, **dict.fromkeys(OG, 0.0))
    for pr, p, ex, nr, _ in v["k"].values():
        r["base"] += pr * G.iou(g132.gate(bs(tmp(p, 0.8)), ex, nr), m)
        for nm, q in og_variants(p, y, sets, mp, mpa).items():
            r[nm] += pr * G.iou(g132.gate(bs(tmp(q, 0.8)), ex, nr), m)
    rows.append(r)
DM = table(rows, OG, "M3 dev ongoing (A17)"); R7 = pd.read_csv(W + "g287_m3.csv").set_index("w")["t0.8"]; print("  identity base against g287:", float(np.abs(DM.set_index("w").base - R7.reindex(DM.w).to_numpy()).max()), flush=True)
D8 = pickle.load(open(W + "g298_on_pretemp.pkl", "rb")); rows = []
for w, v in D8["M"].items():
    m = G.META[w]; mp, mpa = maps(HM[m["panel"]], m["day"], m["t"] % 288); r = dict(w=w, base=0.0, **{k: 0.0 for k in ("O_loc", "O_size", "O_loc_size", "O_map", "O_map_any")})
    for pr, row in v["A"]:
        for nm, s_ in on_sel(row, v["y"], v["e"], mp, mpa).items():
            r[nm] += pr * iou1(s_, v["y"], v["e"])
    rows.append(r)
DO = table(rows, ("O_loc", "O_size", "O_loc_size", "O_map", "O_map_any"), "M3 dev onset (A17)"); P8 = pd.read_csv(W + "g280_m3.csv").set_index("w").pc4; print("  identity base against g280 pc4:", float(np.abs(DO.set_index("w").base - P8.reindex(DO.w).to_numpy()).max()), flush=True)
# --- January / February (capB0) and June (cvf): ongoing through G119's flow, onset from the dumps
PRB = {"2031-01": [pickle.load(open(W + f"probs_jfmh0__capB0_s{s}_fold2030-12.pkl", "rb")) for s in range(4)], "2030-06": [pickle.load(open(W + f"probs_junmh0__cvf_attnglob_s{s}_fold2030-06.pkl", "rb")) for s in (0, 1)]}
PRB["2031-02"] = PRB["2031-01"]
for lab, months in (("January / February", ("2031-01", "2031-02")), ("June 2030", ("2030-06",))):
    rows = []
    for mo in months:
        KS = [k for k in np.flatnonzero(T_.META["is_sel"] & (MON == mo)) if str(T_.META["cond"][k]) == "queue_ongoing" and str(k) in PRB[mo][0]]
        for p in sorted({str(T_.META["panel"][k]) for k in KS}):
            P = h.load(p, "train"); key = (mo, p, "mask"); h.CTX.clear(); h.sim_blank.cache_clear(); h.CTX[key] = h.Ctx(key, P, np.where(P.blank, np.nan, P.V), P.V); c = h.CTX[key]
            rr = np.flatnonzero([s.startswith(mo) for s in P.times]); Hm = head_rows(P.V[rr], P.vcut); r0 = rr[0]
            for k in [k for k in KS if str(T_.META["panel"][k]) == p]:
                t = int(T_.META["t"][k]); dom = int(P.times[t][8:10]) - 1; lab_, el = T_.LAB[k]; y, e = lab_.astype(bool), el.astype(bool)
                lq = np.asarray(h.XN[h.OFFN[k]:h.OFFN[k + 1]][:, h.G.CI], float) == 1; p_ = sum(d[str(k)] for d in PRB[mo]) / len(PRB[mo]); cone = h.cone_parts(c, t, lq, P.length)
                sets = og_sets(lq); mp, mpa = maps(Hm, (t - r0) // 288, t % 288); yt = y & e; r = dict(w=int(k), base=0.0, **dict.fromkeys(OG, 0.0))
                for kk, pr in h.m3v2.kdist(dom, cap=3).items():
                    pc = p_ * h.cone_f(c, cone, t, kk); s_, nd = h.seen(c, t, kk, -18, 12); pb = pc * np.where(h.G.dil(s_, 1), 1.0, 0.85)[None, :] if nd else pc.copy()
                    r["base"] += pr * h.iou(h.far_gate(c, h.decode(tmp(pb, 0.8)), t, kk, lq), y, e, "queue_ongoing")
                    for nm, q in og_variants(pb, yt, sets, mp, mpa).items():
                        r[nm] += pr * h.iou(h.far_gate(c, h.decode(tmp(q, 0.8)), t, kk, lq), y, e, "queue_ongoing")
                rows.append(r)
            h.load.cache_clear()
    Dg = table(rows, OG, f"{lab} ongoing")
    if lab.startswith("January"):
        R8 = pd.read_csv(W + "g287_jf.csv").set_index("w")["t0.8"]; print("  identity base against g287 JF:", float(np.abs(Dg.set_index("w").base - R8.reindex(Dg.w).to_numpy()).max()), flush=True)
    else:
        R9 = pd.read_csv(W + "g321_jun_ongoing.csv").set_index("w")["t0.8"]; print("  identity base against g321:", float(np.abs(Dg.set_index("w").base - R9.reindex(Dg.w).to_numpy()).max()), flush=True)
    src = D8["J"] if lab.startswith("January") else pickle.load(open(W + "g329_jun_onset_rows.pkl", "rb")); rows = []
    for k, v in src.items():
        p = str(T_.META["panel"][k]); t = int(T_.META["t"][k]); mo = MON[k]; P = h.load(p, "train"); rr = np.flatnonzero([s.startswith(mo) for s in P.times]); Hm = head_rows(P.V[rr], P.vcut)
        mp, mpa = maps(Hm, (t - rr[0]) // 288, t % 288); r = dict(w=int(k), base=0.0, **{q: 0.0 for q in ("O_loc", "O_size", "O_loc_size", "O_map", "O_map_any")})
        for pr, row in v["C"]:
            for nm, s_ in on_sel(row, v["y"], v["e"], mp, mpa).items():
                r[nm] += pr * iou1(s_, v["y"], v["e"])
        rows.append(r)
    h.load.cache_clear(); table(rows, ("O_loc", "O_size", "O_loc_size", "O_map", "O_map_any"), f"{lab} onset")
