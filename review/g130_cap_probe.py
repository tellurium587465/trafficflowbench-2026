"""G130 (10/31, ledger #84, strategy P2(a)): score the capacity probe (Kaggle telluriumrrr/tfb-t2-cap1, fold 2 models trained on
six months, seeds 0 / 1, B0 = the q24f_attnglob line). Probabilities = the seed mean.
  m3    : M3 v3 DEV (held-out days not read) from work/t2/probs_m3v3cap__<tag>.pkl (t2_nn_cpu_infer.py): onset = physics onset
          adaptive a + evidence x0.25 + best_set, ongoing = cone + seen x0.85 + best_set + far gate (no X1, no LightGBM).
          Writes work/t2/g130_cap_probe.csv for judge.py --kind model --k 5 --cond ...
  train : 2031-01 / 2031-02 selector-like windows, the out-of-fold probabilities, best_set (raw), both conditions.
usage: TFB_M3VER=v3 py g130_cap_probe.py [m3 | train] [out dir of the kernel]"""
import os, sys, time, pickle, importlib.util
MODE = (sys.argv[1:] or ["m3"])[0]; KO = (sys.argv[2:] or [r"C:/tfb_data/kaggle_out/cap1/out"])[0]
os.environ.setdefault("TFB_M3VER", "v3")
import numpy as np, pandas as pd
TM = time.perf_counter; T00 = TM(); R = r"C:/Users/user/tfb/"
CFGS = ("B0", "D2", "E2", "W2", "W2D2", "W3", "X3b", "X3a"); SEEDS = (0, 1); TAIL = "_fold2030-12"; ONG_ONLY = ("X3a",)
TAG = lambda c: c.lower() if c.startswith("X3") else "cap" + c
if os.environ.get("G130_CFGS"):                                # the columns for a local queue, e.g. B0,L25,L50
    CFGS = tuple(os.environ["G130_CFGS"].split(","))
if os.environ.get("G130_LOCAL"):                               # the local-GPU tags, e.g. G130_LOCAL=B0=b0L,X3b=x3bL,X3a=x3aL G130_SEEDS=1
    _T = dict(x.split("=") for x in os.environ["G130_LOCAL"].split(",")); TAG = lambda c: _T.get(c, "none"); SEEDS = tuple(int(x) for x in os.environ.get("G130_SEEDS", "0,1").split(","))
DRY = os.environ.get("G130_DRY", "") == "1"                     # 5-window dry run: saved probabilities, identity against the saved results


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path); mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


def boot(d, cl, n_boot=2000, seed=0):
    rng = np.random.default_rng(seed); ids = pd.factorize(cl)[0]; M = np.bincount(ids, d); N = np.bincount(ids)
    Rm = rng.multinomial(len(M), np.full(len(M), 1.0 / len(M)), size=n_boot); return float(((Rm @ M) / (Rm @ N)).std())


def seedmean(paths):
    ds = [pickle.load(open(p, "rb")) for p in paths if os.path.exists(p)]
    return ({w: sum(d[w] for d in ds) / len(ds) for w in ds[0] if all(w in d for d in ds)}, len(ds)) if ds else (None, 0)


def m3():
    g = module("g121", R + "review/g121_ongoing_oracle.py"); G, V, m3v2 = g.G, g.V, g.m3v2; PR = {}
    if DRY:
        g2 = module("g120", R + "review/g120_onset_gate.py"); PN = g2.nn_blend(); PO, best_set = g2.PO, g2.best_set; w0 = g.DEV[:5]; n0 = g2.DEV[:5]
        phys = g2.lgb.Booster(model_file=g2.W + "physon_base.txt"); ev = PO.month_events("validation", PO.rarity_table()); ref = pd.read_csv(g.W + "g108_rectangles.csv").set_index("w")["A"]; d0 = 0.0
        for w in n0:
            m = G.META[w]; i = G.POS[w]; Xw = np.asarray(G.X[G.OFF[i]:G.OFF[i + 1]]); ph = phys.predict(PO.feats(Xw)); a = PO.adaptive_a(ev, m["day"]); sc = 0.0
            for k, pr in m3v2.kdist(m["day"]).items():
                pb = PN[w].copy(); pb[5] = (1 - a) * pb[5] + a * ph; s, nd = m3v2.seen(m["panel"], m["t"], k, -6, 12)
                if nd:
                    pb[-1] = pb[-1] * np.where(G.dil(s, 2), 1.0, 0.25)
                sc += pr * G.iou(best_set(pb.ravel(), np.ones(pb.size, bool)).reshape(pb.shape), m)
            d0 = max(d0, abs(sc - ref[w]))
        P0 = pickle.load(open(g.W + "probs_m3v3__q24f_attnglob_s0.pkl", "rb")); print("onset identity (5 windows) max |d|", d0, "| saved prob shape", P0[w0[0]].shape, "| ongoing path", [round(sum(pr * G.iou(g.og_pred(P0[w] * g.cone_f(G.META[w], k), G.META[w], k)[0], G.META[w]) for k, pr in m3v2.kdist(G.META[w]["day"]).items()), 4) for w in w0]); return
    for c in CFGS:
        p, n = seedmean([g.W + f"probs_m3v3cap__{TAG(c)}_s{s}{TAIL}.pkl" for s in SEEDS])
        if p is not None:
            PR[c] = p; print(c, "seeds", n, "windows", len(p), flush=True)
    cols = {c: {} for c in PR}
    for w in g.DEV:
        m = G.META[w]; kd = m3v2.kdist(m["day"]); cf = {k: g.cone_f(m, k) for k in kd}
        for c in PR:
            cols[c][w] = sum(pr * G.iou(g.og_pred(PR[c][w] * cf[k], m, k)[0], m) for k, pr in kd.items())
    print(f"ongoing {TM() - T00:.0f}s", flush=True)
    g2 = module("g120", R + "review/g120_onset_gate.py"); PO, best_set = g2.PO, g2.best_set
    phys = g2.lgb.Booster(model_file=g2.W + "physon_base.txt"); ev = PO.month_events("validation", PO.rarity_table())
    for w in g2.DEV:
        m = G.META[w]; i = G.POS[w]; Xw = np.asarray(G.X[G.OFF[i]:G.OFF[i + 1]]); ph = phys.predict(PO.feats(Xw)); a = PO.adaptive_a(ev, m["day"]); kd = m3v2.kdist(m["day"])
        ss = {k: m3v2.seen(m["panel"], m["t"], k, -6, 12) for k in kd}
        for c in PR:
            if c in ONG_ONLY:
                continue
            sc = 0.0
            for k, pr in kd.items():
                pb = PR[c][w].copy(); pb[5] = (1 - a) * pb[5] + a * ph; s, nd = ss[k]
                if nd:
                    pb[-1] = pb[-1] * np.where(G.dil(s, 2), 1.0, 0.25)
                sc += pr * G.iou(best_set(pb.ravel(), np.ones(pb.size, bool)).reshape(pb.shape), m)
            cols[c][w] = sc
    D = V.frame(cols).dropna(subset=["B0"]); D.to_csv(R + "work/t2/" + os.environ.get("G130_OUT", "g130_cap_probe.csv"), index=False); print(f"m3: {len(D)} windows in {TM() - T00:.0f}s")
    for cond in ("queue_onset", "queue_ongoing"):
        d = D[D.cond == cond]; cl = d.panel + "_" + d.day.astype(str)
        print(cond, "B0", round(d.B0.mean(), 4), " | ".join(f"{c} {(d[c] - d.B0).mean():+.4f} (SE {boot((d[c] - d.B0).to_numpy(), cl):.4f})" for c in PR if c != "B0" and d[c].notna().all()))


def train():
    g = module("g119", R + "review/g119_month_postproc.py"); T = g.T; PR = {}
    if DRY:
        O = [pickle.load(open(g.W + f"oof_q25_nomem_s{s}.pkl", "rb")) for s in (3, 4)]; ref = pd.read_csv(g.W + "g119_month_postproc.csv", dtype={"w": str}); ref = ref[ref.vis == "mask"].set_index(["month", "w"]).raw
        ks = np.flatnonzero(T.META["is_sel"] & np.isin(g.MON, g.MONTHS))[:5]; d0 = 0.0
        for k in ks:
            lab, el = T.LAB[k]; d0 = max(d0, abs(g.iou(g.decode((O[0][k] + O[1][k]) / 2), lab.astype(bool), el.astype(bool), str(T.META["cond"][k])) - ref.loc[(g.MON[k], str(k))]))
        print("train identity (5 windows, raw vs g119) max |d|", d0); return
    for c in CFGS:
        p, n = seedmean([f"{KO}/oof_{TAG(c)}_s{s}.pkl" for s in SEEDS])
        if p is not None:
            PR[c] = p; print(c, "seeds", n, "windows", len(p), flush=True)
    ks = [k for k in np.flatnonzero(T.META["is_sel"] & np.isin(g.MON, g.MONTHS)) if k in PR["B0"]]; rows = []
    for k in ks:
        lab, el = T.LAB[k]; cond = str(T.META["cond"][k]); r = dict(w=int(k), month=g.MON[k], cond=cond, panel=str(T.META["panel"][k]), day=str(T.META["date"][k])[:10])
        for c in PR:
            if k in PR[c] and not (c in ONG_ONLY and cond == "queue_onset"):
                r[c] = g.iou(g.decode(PR[c][k]), lab.astype(bool), el.astype(bool), cond)
        rows.append(r)
    D = pd.DataFrame(rows); D.to_csv(R + "work/t2/g130_cap_probe_train.csv", index=False); print(f"train: {len(D)} windows in {TM() - T00:.0f}s")
    for mo in g.MONTHS:
        for cond in ("queue_onset", "queue_ongoing"):
            d = D[(D.month == mo) & (D.cond == cond)]; cl = d.panel + "_" + d.day
            print(mo, cond.split("_")[1], len(d), "B0", round(d.B0.mean(), 4), " | ".join(
                f"{c} {(d[c] - d.B0).mean():+.4f} (z {(d[c] - d.B0).mean() / max(boot((d[c] - d.B0).to_numpy(), cl), 1e-12):+.1f})" for c in PR if c != "B0" and d[c].notna().all()))


def blend():
    """ledger #86 (2): 0.8 ((1 - w) X1 + w p_X3) + 0.2 LightGBM -> cone -> seen -> best_set -> far gate, M3 dev ongoing, base = G108 column A"""
    g = module("g121", R + "review/g121_ongoing_oracle.py"); G, V, m3v2 = g.G, g.V, g.m3v2; PR = {c: seedmean([g.W + f"probs_m3v3cap__{TAG(c)}_s{s}{TAIL}.pkl" for s in SEEDS])[0] for c in ("X3a", "X3b")}
    cols = {c: {} for c in ["A"] + [f"{c}_w{w}" for c in PR for w in (20, 30)]}
    for wi in g.DEV:
        m = G.META[wi]; i = G.POS[wi]; Xw = np.asarray(G.X[G.OFF[i]:G.OFF[i + 1]]); pl = g.FS.lgb_ongoing(Xw); sc = {c: 0.0 for c in cols}
        for k, pr in m3v2.kdist(m["day"]).items():
            cf = g.cone_f(m, k); x1 = g.X1[(wi, k)]; sc["A"] += pr * G.iou(g.og_pred((0.8 * x1 + 0.2 * pl) * cf, m, k)[0], m)
            for c in PR:
                for w in (20, 30):
                    sc[f"{c}_w{w}"] += pr * G.iou(g.og_pred((0.8 * ((1 - w / 100) * x1 + w / 100 * PR[c][wi]) + 0.2 * pl) * cf, m, k)[0], m)
        for c in sc:
            cols[c][wi] = sc[c]
    D = V.frame(cols).dropna(subset=["A"]); ref = pd.read_csv(g.W + "g108_rectangles.csv").set_index("w")["A"]; print("identity A vs G108", float(np.max(np.abs(D.A - ref.loc[D.w].to_numpy()))))
    D.to_csv(R + "work/t2/g130_x3_blend.csv", index=False); cl = D.panel + "_" + D.day.astype(str)
    for c in cols:
        if c != "A":
            x = (D[c] - D.A).to_numpy(); se = boot(x, cl); print(f"{c}: {x.mean():+.4f} (SE {se:.4f}, z {x.mean() / se:+.2f}, /1.25 {x.mean() / se / 1.25:+.2f})")


if __name__ == "__main__":
    {"m3": m3, "train": train, "blend": blend}[MODE]()
