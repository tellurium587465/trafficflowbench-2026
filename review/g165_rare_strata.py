"""G165 (11/01, ledger #119, addendum of step 1, no line of its own): G164 found March's month start rich in rare-site births, and the official onset windows sit there (March 39 of 40, April 35 of 40).
(1) births at a rare site by day group in April (the released layer, nothing else). (2) onset components on M3 dev by stratum: rare site or usual site (G117 fmax <= 0.1), evidence x0.25 (E - raw),
physics onset (PE - E), both (PE - raw), on G119's March rows; then the same read at a rare-site share of 0.26 (M3 dev), 0.40 (M4b), 0.48 (March month start, G164). The reweighting of the two strata is a mix of the
two means, its SE from the two strata. Hold-out days are not read.
usage: TFB_M3VER=v3 py -B -X utf8 g165_rare_strata.py"""
import sys
R = r"C:/Users/user/tfb/review/"; sys.dont_write_bytecode = True
src = open(R + "g119_month_postproc.py", encoding="utf-8").read().split('if __name__ == "__main__":')[0]
sys.argv = [sys.argv[0], "run"]; exec(compile(src, "g119", "exec"), globals())
GR = ("1-5", "6-10", "11+"); grp = lambda d: np.select([d <= 4, d <= 9], [0, 1], 2)


def clm(d, v, n_boot=2000, seed=0):
    ids = pd.factorize(d.panel + "_" + d.day.astype(str))[0]; a = np.asarray(v, float); M = np.bincount(ids, a); N = np.bincount(ids); Rm = np.random.default_rng(seed).multinomial(len(M), np.full(len(M), 1 / len(M)), size=n_boot); return a.mean(), float(((Rm @ M) / (Rm @ N)).std())


F0 = PO.rarity_table(); rows = []
for p in sorted({G.META[w]["panel"] for w in G.WID}):
    P = load(p, "private"); key = ("pri", p); CTX.clear(); CTX[key] = Ctx(key, P, P.V); c = CTX[key]
    for d_, rare in month_events(c, p, P, "2031-04", F0):
        rows.append(dict(panel=p, day=d_, rare=rare))
    load.cache_clear()
E = pd.DataFrame(rows); E["g"] = grp(E.day.to_numpy())
print("April births")
for j, g_ in E.groupby("g"):
    m_, s_ = clm(g_, g_.rare); print(f"  day {GR[j]:5s}: births {len(g_):4d}, rare {m_:.3f} (SE {s_:.3f})")
a, b = E[E.g == 0], E[E.g > 0]; (ra, sa), (rb, sb) = clm(a, a.rare), clm(b, b.rare); print(f"  1-5 minus the rest {ra - rb:+.3f} (SE {np.hypot(sa, sb):.3f}, z {(ra - rb) / np.hypot(sa, sb):+.2f})", flush=True)
D = pd.read_csv(W + "g119_month_postproc.csv", dtype={"w": str}); D = D[(D.month == "2031-03") & (D.cond == "queue_onset")].copy(); O = pd.read_csv(W + "g117_onset_oracle.csv").set_index("w"); D["rare"] = O.fmax.reindex(D.w).to_numpy() <= 0.1; D = D[O.nY.reindex(D.w).to_numpy() > 0]
print(f"M3 dev onset with a label {len(D)}, rare {int(D.rare.sum())}; IoU PE rare {D[D.rare].PE.mean():.4f} usual {D[~D.rare].PE.mean():.4f}")
for nm, a_, b_ in (("evidence x0.25 (E - raw)", "raw", "E"), ("physics (PE - E)", "E", "PE"), ("both (PE - raw)", "raw", "PE")):
    (mr, sr), (mu, su) = clm(D[D.rare], D[D.rare][b_] - D[D.rare][a_]), clm(D[~D.rare], D[~D.rare][b_] - D[~D.rare][a_]); out = [f"{nm:26s} rare {mr:+.4f} (SE {sr:.4f}) | usual {mu:+.4f} (SE {su:.4f})"]
    for s_ in (0.26, 0.40, 0.48):
        out.append(f"at {s_:.2f}: {s_ * mr + (1 - s_) * mu:+.4f} (SE {np.hypot(s_ * sr, (1 - s_) * su):.4f})")
    print(" | ".join(out), flush=True)
