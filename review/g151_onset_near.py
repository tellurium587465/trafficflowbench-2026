"""G151 (11/01, ledger #111, stage 0, literature audit): where do the onset misses land, and are the neighbours of the top link calibrated?
Top link j = argmax of the network's T+30 probability. Y = the links queued at T+30 on eligible cells. Link numbers grow downstream.
  (a) windows with j not in Y: signed distance from j to the nearest link of Y. Windows with j in Y are left out: Y runs upstream from
      its head, so a distance to the head over all windows leans one way by construction.
  (b) for s in -2, -1, +1, +2: over windows where j+s is eligible, the share queued against the mean network probability at j+s,
      window bootstrap of the ratio.
Windows: M3 dev onset with a precursor (g117 flags, the held-out days are not read); January and February onset windows, month-CV OOF
of q25_nomem seeds 3, 4. Raw network probabilities, before the evidence rule and the ranker.
usage: TFB_M3VER=v3 py -B -X utf8 g151_onset_near.py"""
import os, sys, pickle
R = r"C:/Users/user/tfb/review/"; sys.dont_write_bytecode = True
src = open(R + "g119_month_postproc.py", encoding="utf-8").read().split('if __name__ == "__main__":')[0]
sys.argv = [sys.argv[0], "run"]; exec(compile(src, "g119", "exec"), globals())
import importlib.util
sp = importlib.util.spec_from_file_location("g120", R + "g120_onset_gate.py"); g2 = importlib.util.module_from_spec(sp); sp.loader.exec_module(g2)
PN = g2.nn_blend(); Om = pd.read_csv(W + "g117_onset_oracle.csv").set_index("w")
nop = (Om.v_ff >= 0.98) & Om.flow_ratio.between(0.85, 1.15) & (Om.fmax <= 0.1)
print("M3 dev onset", len(Om), "no precursor", int(nop.sum()), flush=True)
OOF = [pickle.load(open(W + f"oof_q25_nomem_s{s}.pkl", "rb")) for s in (3, 4)]


def one(p5, y5, e5, tag, w):
    Y = np.flatnonzero(y5 & e5); j = int(np.argmax(p5)); rows = []
    if len(Y) == 0:
        return None, rows
    d = None if j in set(Y.tolist()) else int(min(Y - 0, key=lambda x: (abs(j - x), x)) * -1 + j)
    for s in (-2, -1, 1, 2):
        i = j + s
        if 0 <= i < len(p5) and e5[i]:
            rows.append(dict(month=tag, w=w, s=s, obs=float(y5[i]), pred=float(p5[i]), hit=j in set(Y.tolist())))
    return d, rows


D, N = [], []
for w in g2.DEV:
    m = G.META[w]
    if m["cond"] != "queue_onset" or bool(nop.get(w, False)):
        continue
    y, e = m["y"].T.astype(bool), m["m"].T.astype(bool); d, r = one(np.asarray(PN[w])[5], y[5], e[5], "2031-03", w); N += r
    D.append(dict(month="2031-03", w=str(w), d=d, top=float(np.max(np.asarray(PN[w])[5]))))
for mo in MONTHS:
    for k in np.flatnonzero(T.META["is_sel"] & (MON == mo)):
        if str(T.META["cond"][k]) != "queue_onset":
            continue
        lab, el = T.LAB[k]; p = (OOF[0][k] + OOF[1][k]) / 2; d, r = one(p[5], lab[5].astype(bool), el[5].astype(bool), mo, int(k)); N += r
        D.append(dict(month=mo, w=str(k), d=d, top=float(p[5].max())))
D = pd.DataFrame(D); N = pd.DataFrame(N); D.to_csv(W + "g151_onset_near.csv", index=False)
print("\n(a) misses: signed distance from the top link to the nearest true link (+ = the top link is downstream of the queue)")
for mo, g in D.groupby("month"):
    x = g.d.dropna().to_numpy(); nz = x[x != 0]; up, dn = int((nz < 0).sum()), int((nz > 0).sum())
    print(f"  {mo}: windows {len(g)}, misses {len(x)} ({len(x) / len(g):.1%}) | upstream {up}, downstream {dn}, one-sided share {max(up, dn) / max(len(nz), 1):.0%} | |d| = 1: {int((abs(x) == 1).sum())}, 2: {int((abs(x) == 2).sum())}, >= 3: {int((abs(x) >= 3).sum())}")
print("\n(b) neighbours of the top link: observed share against the mean network probability")


def ratio(g, n=2000, seed=0):
    r = np.random.default_rng(seed); ids = pd.factorize(g.w)[0]; O = np.bincount(ids, g.obs); P = np.bincount(ids, g.pred); Rm = r.multinomial(len(O), np.full(len(O), 1 / len(O)), size=n)
    v = (Rm @ O) / (Rm @ P); return O.sum() / P.sum(), np.quantile(v, .025), np.quantile(v, .975)


TR = N[N.month.isin(MONTHS)]
for nm, g in (("January + February", TR), ("2031-01", N[N.month == "2031-01"]), ("2031-02", N[N.month == "2031-02"]), ("March dev (precursor)", N[N.month == "2031-03"])):
    print(" ", nm)
    for lab_, h in (("s = +1", g[g.s == 1]), ("s = -1", g[g.s == -1]), ("s = +-1", g[g.s.abs() == 1]), ("s = +-2", g[g.s.abs() == 2]), ("s = +-1, top missed", g[(g.s.abs() == 1) & (~g.hit)])):
        if len(h):
            a, lo, hi = ratio(h); print(f"    {lab_:20s} cells {len(h):5d} observed {h.obs.mean():.4f} network {h.pred.mean():.4f} ratio {a:.2f} [{lo:.2f}, {hi:.2f}]")
