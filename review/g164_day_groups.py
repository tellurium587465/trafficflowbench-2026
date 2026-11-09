"""G164 (11/01, ledger #119 step 1, strategy T2_DEEP_REVIEW 6-1): the official windows sit in the first days of the month and M3 dev across the whole month. The number of earlier days is
already simulated (k); a month-start mix of incidents is not. Per calendar-day group (1-5, 6-10, 11+): final A's IoU on M3 dev (g155, g156), rare-site and no-precursor shares of the onset
windows, and the share of queue births at a rare site on the dev days of March and on all nine training months (G119's month_events, rarity without the month). The level is also raked
to the official day shares on top of the March composition. Hold-out days are not read.
usage: TFB_M3VER=v3 py -B -X utf8 g164_day_groups.py"""
import sys
R = r"C:/Users/user/tfb/review/"; sys.dont_write_bytecode = True
src = open(R + "g119_month_postproc.py", encoding="utf-8").read().split('if __name__ == "__main__":')[0]
sys.argv = [sys.argv[0], "run"]; exec(compile(src, "g119", "exec"), globals())
GR = ("1-5", "6-10", "11+"); grp = lambda d: np.select([d <= 4, d <= 9], [0, 1], 2)               # d = 0-based day
ON = pd.read_csv(W + "g155_onset_b_m3.csv"); OG = pd.read_csv(W + "g156_m3_frame.csv"); O117 = pd.read_csv(W + "g117_onset_oracle.csv").set_index("w"); ON["rare"] = O117.fmax.reindex(ON.w).to_numpy() <= 0.1


def clm(d, v, n_boot=2000, seed=0):
    ids = pd.factorize(d.panel + "_" + d.day.astype(str))[0]; a = np.asarray(v, float); M = np.bincount(ids, a); N = np.bincount(ids); Rm = np.random.default_rng(seed).multinomial(len(M), np.full(len(M), 1 / len(M)), size=n_boot); return a.mean(), float(((Rm @ M) / (Rm @ N)).std())


for nm, D, col in (("onset", ON, "A_u"), ("ongoing", OG, "A")):
    g_ = grp(D.day.to_numpy())
    for j, lab in enumerate(GR):
        d = D[g_ == j]
        if len(d):
            m_, s_ = clm(d, d[col]); print(f"{nm:8s} day {lab:5s}: windows {len(d):3d}, IoU {m_:.4f} (SE {s_:.4f})" + (f", rare site {d.rare.mean():.3f}, no precursor (O) {d.flag_O.mean():.3f}" if nm == "onset" else ""))
    a, b = D[g_ == 0], D[g_ > 0]; (ma, sa), (mb, sb) = clm(a, a[col]), clm(b, b[col]); print(f"{nm:8s} 1-5 minus the rest: {ma - mb:+.4f} (SE {np.hypot(sa, sb):.4f}, z {(ma - mb) / np.hypot(sa, sb):+.2f})", flush=True)
    if nm == "onset":
        (ra, sa), (rb, sb) = clm(a, a.rare), clm(b, b.rare); print(f"         rare share 1-5 minus the rest: {ra - rb:+.3f} (SE {np.hypot(sa, sb):.3f}, z {(ra - rb) / np.hypot(sa, sb):+.2f})")
ev = PO.month_events("validation", PO.rarity_table()); print(type(ev), (list(ev.items())[:1] if isinstance(ev, dict) else ev[:2]), flush=True)
rows = []
F0 = PO.rarity_table()
for p in sorted({G.META[w]["panel"] for w in G.WID}):
    P = load(p, "validation"); key = ("val", p); CTX[key] = Ctx(key, P, P.V); c = CTX[key]
    for d_, rare in month_events(c, p, P, "2031-03", F0):
        if not m3v2.is_holdout(np.array([d_]))[0]:
            rows.append(dict(month="2031-03", panel=p, day=d_, rare=rare))
    load.cache_clear()
    P = load(p, "train"); Vm = np.where(P.blank, np.nan, P.V)
    for mo in np.unique([str(t)[:7] for t in P.times]):
        F_ = rarity_lomo(mo); key = (mo, p); CTX.clear(); CTX[key] = Ctx(key, P, Vm, P.V); c = CTX[key]
        for d_, rare in month_events(c, p, P, mo, F_):
            rows.append(dict(month=mo, panel=p, day=d_, rare=rare))
    load.cache_clear(); print(p, len(rows), flush=True)
E = pd.DataFrame(rows); E.to_csv(W + "g164_births.csv", index=False); E["g"] = grp(E.day.to_numpy()); E["tr"] = E.month != "2031-03"
print("births per day and rare share by day group")
for tr, lab in ((False, "March dev days"), (True, "nine training months")):
    e = E[E.tr == tr]; nd = {0: 5, 1: 5, 2: 20}
    for j, g_ in e.groupby("g"):
        m_, s_ = clm(g_, g_.rare); print(f"  {lab:22s} day {GR[j]:5s}: births {len(g_):5d}, rare {m_:.3f} (SE {s_:.3f})")
    a, b = e[e.g == 0], e[e.g > 0]; (ra, sa), (rb, sb) = clm(a, a.rare), clm(b, b.rare); print(f"  {lab:22s} 1-5 minus the rest: {ra - rb:+.3f} (SE {np.hypot(sa, sb):.3f}, z {(ra - rb) / np.hypot(sa, sb):+.2f})", flush=True)
for nm, D, col, sh in (("onset", ON, "A_u", {0: 39 / 40, 1: 1 / 40, 2: 0.0}), ("ongoing", OG, "A", {0: 30 / 40, 1: 6 / 40, 2: 4 / 40})):
    x = D.assign(dg=grp(D.day.to_numpy())); w0, e0 = m3v2.weights(x, "validation"); w1, e1 = m3v2.weights(x, "validation", extra={"queue_" + nm: [("dg", sh)]}); a = x[col].to_numpy()
    print(f"{nm}: March composition {(a * w0).sum() / w0.sum():.4f} (ESS {list(e0.values())[0]:.0f}) -> plus the official day shares {(a * w1).sum() / w1.sum():.4f} (ESS {list(e1.values())[0]:.0f})")
