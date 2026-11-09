"""G131 (10/31, ledger #85, strategy P2(b)): a short CTM rollout from the state at T-5 as an ongoing post-processing step.
State = last_v / last_q of the network inputs (density by link, gaps interpolated), twin/ctm.py Corridor (lmin 0.2, released
FD), 35 min ahead, upstream = the first 3 links' q_m3 held, junction ratios by time of day from the training months (the month
left out on January / February). C0 = as specified, C1 = the head cell of each jam at T-5 keeps its observed flow as capacity
(the FD alone has no incident, so a queue dissolves). Capacity x (1.0, 1.05, 1.1).
Factor per row: a jam [a, b] at T-5 that the rollout loses gets x0.7; otherwise links more than 2 upstream of the rollout's tail
(up to 12, not past the previous jam's head) get x0.5; same-month evidence exempts a link as in the cone. After the cone,
before seen x0.85, best_set and the far gate.
  m3    : M3 v3 DEV ongoing 356 windows, base = G108 column A, frame for judge.py, J15-type split, skill of the rollout alone
  train : 2031-01 / 2031-02 selector-like ongoing windows, G119 path, base = SCG
usage: TFB_M3VER=v3 py g131_ctm_roll.py [m3 | train]   (G131_DRY=1: a few windows, identity only)"""
import os, sys, time, pickle, importlib.util
from collections import defaultdict
MODE = (sys.argv[1:] or ["m3"])[0]; DRY = os.environ.get("G131_DRY", "") == "1"
os.environ.setdefault("TFB_M3VER", "v3"); R0 = r"C:/Users/user/tfb/"
sys.path.insert(0, R0 + "src"); sys.path.insert(0, R0 + "twin")
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from ctm import Corridor, junction_ratios
from common import NN
TM = time.perf_counter; T00 = TM(); OUT = R0 + "work/t2/ctm_roll/"; os.makedirs(OUT, exist_ok=True); LMIN = 0.2
CAPS = (1.0, 1.05, 1.1); VARS = [f"{c}_{int(m * 100)}" for c in ("C0", "C1") for m in CAPS]
NAMES = [str(x) for x in np.load(f"{NN}/nn_data_afdem/names.npy", allow_pickle=True)]; IX = {n: i for i, n in enumerate(NAMES)}


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path); mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


def runs(v):
    dd = np.diff(np.r_[0, np.asarray(v, int), 0]); return list(zip(np.flatnonzero(dd == 1), np.flatnonzero(dd == -1) - 1))


def roll(C, k0, up, CR, hc):
    """k0 (nd, NC) at the end of T-5, up (nd,) vph, CR (nd, 7, NC), hc (nd, NC) capacity caps -> (nd, 6, NC) mean speed, T+5..T+30"""
    nd = k0.shape[0]; dt = C.dt; k = k0.copy(); qin = np.zeros(nd); V = np.zeros((nd, 6, C.NC)); cap = np.minimum(C.cap[None, :], hc)
    for r in range(7):
        crj = CR[:, r, :-1]; fa = np.zeros((nd, C.NC)); ka = np.zeros((nd, C.NC))
        for _ in range(C.sub):
            S = np.minimum(C.vf * k, cap); Rc = np.minimum(cap, C.w * np.maximum(C.kj - k, 0))
            f = np.minimum(S[:, :-1], Rc[:, 1:] / np.maximum(crj, 1e-3))
            inf0 = np.minimum(up + qin / dt, Rc[:, 0]); qin = np.maximum(qin + (up - inf0) * dt, 0)
            inn = np.concatenate([inf0[:, None], crj * f], axis=1); out = np.concatenate([f, S[:, -1:]], axis=1)
            k = np.maximum(k + dt / C.cl * (inn - out), 0); fa += out; ka += k
        if r:
            kk = ka / C.sub; V[:, r - 1] = np.where(kk > 1e-3, np.minimum(fa / C.sub / np.maximum(kk, 1e-6), C.vf), C.vf)
    return V


def rollouts(PT, items, rows=None):
    """items: list of (key, Xw, t, weekend) of one panel -> {key: {variant: (6, n) bool}}"""
    n = PT.n; RAT = junction_ratios(PT, rows); out = {it[0]: {} for it in items}; L = np.nan_to_num(PT.length, nan=0.1)
    for m_ in CAPS:
        C = Corridor(PT, lmin=LMIN, cap_mult=m_); cell = C.cell_of; nd = len(items)
        k0 = np.zeros((nd, C.NC)); up = np.zeros(nd); CR = np.ones((nd, 7, C.NC)); hc = np.full((nd, C.NC), np.inf)
        for i, (key, Xw, t, we) in enumerate(items):
            lv, lq_, qm = (np.asarray(Xw[:, IX[c]], float) for c in ("last_v", "last_q", "q_m3"))
            v = lv * PT.vf; q = lq_ * PT.cap
            with np.errstate(all="ignore"):
                k = q / np.maximum(v, 1.0)
            ok = np.isfinite(k); xi = np.arange(n); k = np.where(ok, k, np.interp(xi, xi[ok], k[ok])) if ok.sum() >= 2 else 0.5 * PT.cap / PT.vf
            k0[i] = np.clip(np.bincount(cell, k * L, C.NC) / np.maximum(C.cl, 1e-6), 0, C.kj)
            qq = qm[:3] * PT.cap[:3]; up[i] = float(np.nanmean(qq)) if np.isfinite(qq).any() else C.cap[0] * 0.5
            CR[i] = C.cell_ratios(RAT[int(we), (t - 1 + np.arange(7)) % 288])
            q0 = np.asarray(Xw[:, IX["last_queue"]], float) == 1
            for a, b in runs(q0):
                qb = qm[b] * PT.cap[b]; hc[i, cell[b]] = min(hc[i, cell[b]], max(qb if np.isfinite(qb) else C.cap[cell[b]], 0.2 * C.cap[cell[b]]))
        for tag, h in (("C0", np.full((nd, C.NC), np.inf)), ("C1", hc)):
            V = roll(C, k0, up, CR, h)[:, :, cell]
            for i, it in enumerate(items):
                out[it[0]][f"{tag}_{int(m_ * 100)}"] = V[i] <= PT.vcut[None, :]
    return out


def factor(Rw, q0, ex):
    f = np.ones(Rw.shape); rs = runs(q0)
    for j in range(6):
        prev = -1
        for a, b in rs:
            seg = Rw[j, a:b + 1]
            if not seg.any():
                f[j, a:b + 1] *= 0.7
            else:
                t_ = a + int(np.argmax(seg))
                while t_ > 0 and Rw[j, t_ - 1]:
                    t_ -= 1
                lo, hi = max(prev + 1, t_ - 12), t_ - 3
                if hi >= lo:
                    f[j, lo:hi + 1] *= 0.5
            prev = b
    return np.where(ex[None, :], 1.0, f)


def boot(d, cl, n_boot=2000, seed=0):
    rng = np.random.default_rng(seed); ids = pd.factorize(cl)[0]; M = np.bincount(ids, d); N = np.bincount(ids)
    Rm = rng.multinomial(len(M), np.full(len(M), 1.0 / len(M)), size=n_boot); return float(((Rm @ M) / (Rm @ N)).std())


def m3():
    g = module("g121", R0 + "review/g121_ongoing_oracle.py"); G, V, m3v2 = g.G, g.V, g.m3v2
    from cachelib import load
    DEV = g.DEV[:5] if DRY else g.DEV; by = defaultdict(list)
    for w in DEV:
        m = G.META[w]; i = G.POS[w]; by[m["panel"]].append((w, np.asarray(G.X[G.OFF[i]:G.OFF[i + 1]]), m["t"], (m3v2.MARCH_FIRST_DOW + int(m["day"])) % 7 >= 5))
    t0 = TM(); RO = {}
    for p, items in by.items():
        RO.update(rollouts(load(p, "train"), items)); load.cache_clear()
    print(f"rollouts {len(RO)} windows x {len(VARS)} in {TM() - t0:.1f}s", flush=True)
    cols = {c: {} for c in ["A"] + VARS}; parts = {c: defaultdict(float) for c in ["A"] + VARS}; rows = []; aucy, aucs = [], defaultdict(list)
    for w in DEV:
        m = G.META[w]; i = G.POS[w]; Xw = np.asarray(G.X[G.OFF[i]:G.OFF[i + 1]]); pl = g.FS.lgb_ongoing(Xw); q0 = np.asarray(Xw[:, G.CI], float) == 1; q0d = G.dil(q0, 1)
        y, e = m["y"].T.astype(bool), m["m"].T.astype(bool); sc = defaultdict(float)
        P, _, q = m3v2._panel(m["panel"]); t = m["t"]; d0_ = t - (t % 288) + 60; _, cone, _ = g.g55.parts(w)
        for k, pr in m3v2.kdist(m["day"]).items():
            hid = V._hidden_today(m, k); rr = [r for r in range(d0_, t) if r not in hid]; ev_ = q[rr].any(axis=0) if rr else np.zeros(P.n, bool); s2, _ = m3v2.seen(m["panel"], t, k, -18, 12)
            ex = G.dil(ev_ | s2, 2); cf = np.where(ex[None, :], 1.0, cone); p = (0.8 * g.X1[(w, k)] + 0.2 * pl) * cf
            sel, _ = g.og_pred(p, m, k); sc["A"] += pr * G.iou(sel, m); pa, _ = g.decompose(sel, m, q0d)
            for c_, v_ in pa.items():
                parts["A"][c_.split("_")[0]] += pr * v_ / len(DEV)
            for c in VARS:
                sel2, _ = g.og_pred(p * factor(RO[w][c], q0, ex), m, k); sc[c] += pr * G.iou(sel2, m); pb, _ = g.decompose(sel2, m, q0d)
                for c_, v_ in pb.items():
                    parts[c][c_.split("_")[0]] += pr * v_ / len(DEV)
        for c in sc:
            cols[c][w] = sc[c]
        r = dict(w=w, persist=G.iou(np.broadcast_to(q0, y.shape), m))
        for c in ("C0_100", "C1_100"):
            r[c] = G.iou(RO[w][c], m)
        rows.append(r)
        for a, b in runs(q0):
            ob = e[5, a:b + 1]
            if ob.any():
                aucy.append(int(not (y[5, a:b + 1] & ob).any()))
                for c in ("C0_100", "C1_100"):
                    aucs[c].append(1.0 - float(RO[w][c][5, a:b + 1].mean()))
    ref = pd.read_csv(G.W + "g108_rectangles.csv").set_index("w")["A"]; da = max(abs(cols["A"][w] - ref[w]) for w in DEV)
    print(f"identity base vs G108 column A max |d| {da:.2e} | {TM() - T00:.0f}s", flush=True)
    if da > 1e-9:
        print("IDENTITY FAILED: stop"); sys.exit(1)
    S = pd.DataFrame(rows)
    print("rollout alone (IoU, M3 dev ongoing):", S[["persist", "C0_100", "C1_100"]].mean().round(4).to_dict(), "| gone-at-T+30 AUC:",
          {c: round(roc_auc_score(aucy, aucs[c]), 3) for c in aucs} if len(set(aucy)) > 1 else None, f"(runs {len(aucy)}, gone {np.mean(aucy):.3f})")
    if DRY:
        return
    D = V.frame(cols).dropna(subset=["A"] + VARS); D.to_csv(OUT + "g131_ctm_roll_m3.csv", index=False)
    pd.DataFrame(parts).T.to_csv(OUT + "g131_parts.csv"); cl = D.panel + "_" + D.day.astype(str)
    odd = (D.day % 2 == 1).to_numpy(); best = max(VARS, key=lambda c: (D[c] - D.A)[odd].mean())
    for c in VARS:
        x = (D[c] - D.A).to_numpy(); se = boot(x, cl); print(f"{c}: dev {x.mean():+.4f} (SE {se:.4f}, z {x.mean() / max(se, 1e-12):+.1f}) | odd {x[odd].mean():+.4f} even {x[~odd].mean():+.4f} | windows changed {(np.abs(x) > 1e-9).sum()}")
    x = (D[best] - D.A)[~odd].to_numpy(); print(f"selected on odd days: {best}; even days {x.mean():+.4f} (SE {boot(x, cl[~odd]):.4f})")
    print("loss parts:", {c: {k_: round(v, 4) for k_, v in parts[c].items()} for c in ("A", best)})


def train():
    g = module("g119", R0 + "review/g119_month_postproc.py"); T, G, m3v2 = g.T, g.G, g.m3v2
    OOF = [pickle.load(open(g.W + f"oof_q25_nomem_s{s}.pkl", "rb")) for s in (3, 4)]; rows = []
    for mo in g.MONTHS:
        ks = np.flatnonzero(T.META["is_sel"] & (g.MON == mo) & np.array([str(c) == "queue_ongoing" for c in T.META["cond"]]))
        for p in sorted({str(T.META["panel"][k]) for k in ks}):
            P = g.load(p, "train"); key = (mo, p, "mask"); g.CTX.clear(); g.sim_blank.cache_clear()
            g.CTX[key] = g.Ctx(key, P, np.where(P.blank, np.nan, P.V), P.V); c = g.CTX[key]; kp = [k for k in ks if str(T.META["panel"][k]) == p]
            if DRY:
                kp = kp[:2]
            items = [(int(k), np.asarray(g.XN[g.OFFN[k]:g.OFFN[k + 1]]), int(T.META["t"][k]), pd.Timestamp(P.times[int(T.META["t"][k])][:10]).dayofweek >= 5) for k in kp]
            RO = rollouts(P, items, ~np.char.startswith(P.times.astype(str), mo))
            for (k, Xw, t, we) in items:
                lab, el = T.LAB[k]; y, e = lab.astype(bool), el.astype(bool); lq = np.asarray(Xw[:, G.CI], float) == 1; p_ = (OOF[0][k] + OOF[1][k]) / 2; dom = int(P.times[t][8:10]) - 1
                cone = g.cone_parts(c, t, lq, P.length); r = defaultdict(float, w=k, month=mo, panel=p, day=dom)
                for kk, pr in m3v2.kdist(dom, cap=3).items():
                    ev_ = g.today_ev(c, t, kk); s2, _ = g.seen(c, t, kk, -18, 12); ex = G.dil(ev_ | s2, 2); pc = p_ * np.where(ex[None, :], 1.0, cone)
                    r["A"] += pr * g.iou(g.ongoing_pred(c, pc, t, kk, True, lq), y, e, "queue_ongoing")
                    for v in VARS:
                        r[v] += pr * g.iou(g.ongoing_pred(c, pc * factor(RO[k][v], lq, ex), t, kk, True, lq), y, e, "queue_ongoing")
                rows.append(dict(r))
            g.load.cache_clear()
        print(mo, "done", len(rows), f"{TM() - T00:.0f}s", flush=True)
        if DRY:
            break
    D = pd.DataFrame(rows); ref = pd.read_csv(g.W + "g119_month_postproc.csv", dtype={"w": str}); ref = ref[(ref.vis == "mask") & (ref.cond == "queue_ongoing")].set_index(["month", "w"]).SCG
    da = float(np.max(np.abs(D.A.to_numpy() - np.array([ref.loc[(r.month, str(r.w))] for r in D.itertuples()])))); print(f"identity base vs G119 SCG max |d| {da:.2e} | {TM() - T00:.0f}s", flush=True)
    if da > 1e-9:
        print("IDENTITY FAILED: stop"); sys.exit(1)
    if DRY:
        return
    D.to_csv(OUT + "g131_ctm_roll_train.csv", index=False)
    for mo in g.MONTHS:
        d = D[D.month == mo]; cl = d.panel + "_" + d.day.astype(str); print(mo, len(d), "base", round(d.A.mean(), 4), " | ".join(f"{c} {(d[c] - d.A).mean():+.4f} (z {(d[c] - d.A).mean() / max(boot((d[c] - d.A).to_numpy(), cl), 1e-12):+.1f})" for c in VARS))


if __name__ == "__main__":
    {"m3": m3, "train": train}[MODE]()
