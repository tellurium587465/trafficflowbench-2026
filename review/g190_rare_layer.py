"""G190 (11/01, ledger #158, strategy DISCOVERY_ANGLES, no line): the components read inside and outside the distribution. Layers by how unusual the window's place is for the months that trained the model. Onset: rare site (G117 / rarity without the month). Ongoing: the links queued at T, their queue share in the training months at that clock (other months for January, February), the rarest quarter of the windows against the rest.
Columns from saved frames: M3 dev (g156, g181, g170) and January, February (g119, g181). A table of the signs. Hold-out days are not read.
usage: TFB_M3VER=v3 py -B -X utf8 g190_rare_layer.py"""
import sys, pickle
R = r"C:/Users/user/tfb/review/"; sys.dont_write_bytecode = True
src = open(R + "g119_month_postproc.py", encoding="utf-8").read().split('if __name__ == "__main__":')[0]
sys.argv = [sys.argv[0], "run"]; exec(compile(src, "g119", "exec"), globals())


def clm(d, v, n_boot=1000):
    ids = pd.factorize(d.panel + "_" + d.day.astype(str))[0]; a = np.asarray(v, float); M = np.bincount(ids, a); N = np.bincount(ids); Rm = np.random.default_rng(0).multinomial(len(M), np.full(len(M), 1 / len(M)), size=n_boot); return a.mean(), float(((Rm @ M) / (Rm @ N)).std())


SH = {}


def share(p, excl):
    if (p, excl) not in SH:
        P = load(p, "train"); m = ~np.char.startswith(P.times.astype(str), excl) if excl else np.ones(P.T, bool); V = P.V[m]; q = (V <= P.vcut[None, :]) & np.isfinite(V); SH[(p, excl)] = q.mean(axis=0) + 1e-5
    return SH[(p, excl)]


Fg = pd.read_csv(W + "g156_m3_frame.csv"); rows = {}
for w in Fg.w:
    m = G.META[w]; i = G.POS[w]; lq = np.asarray(G.X[G.OFF[i]:G.OFF[i + 1], G.CI], float) == 1; rows[w] = float(np.log(share(m["panel"], None)[lq]).mean()) if lq.any() else 0.0
Fg["s"] = Fg.w.map(rows); cut = Fg.s.quantile(0.25); Fg["rare"] = Fg.s <= cut
O = pd.read_csv(W + "g119_month_postproc.csv", dtype={"w": str}); O = O[(O.vis == "mask") & (O.cond == "queue_ongoing")].copy(); X = pd.read_csv(W + "g181_train_og.csv", dtype={"w": str}).set_index(["month", "w"]); rr = {}
for r in O.itertuples():
    k = int(r.w); lab_ = np.asarray(XN[OFFN[k]:OFFN[k + 1], G.CI], float) == 1; rr[(r.month, r.w)] = float(np.log(share(r.panel, r.month)[lab_]).mean()) if lab_.any() else 0.0
O["s"] = [rr[(r.month, r.w)] for r in O.itertuples()]; O["rare"] = O.groupby("month").s.transform(lambda x: x <= x.quantile(0.25)); O["x1"] = [X.X1.get((r.month, r.w), np.nan) - X.X0.get((r.month, r.w), np.nan) for r in O.itertuples()]
print("ongoing: rarest quarter against the rest")
print("component | M3 rare | M3 rest | train rare | train rest")
rows_ = [("far gate", Fg.A - Fg.no_gate, O.SG - O.S), ("cone", Fg.A - Fg.no_cone, O.SCG - O.SG), ("X1", Fg.A - Fg.no_x1, O.x1), ("seen x0.85", None, O.S - O.raw), ("LightGBM", Fg.A - Fg.no_lgb, None), ("tfm", Fg.A - Fg.no_tfm, None)]
for nm, a, b in rows_:
    o = [nm]
    if a is not None:
        for m_ in (Fg.rare, ~Fg.rare):
            d = Fg[m_]; v, s = clm(d, a[m_]); o.append(f"{v:+.4f} ({s:.4f})")
    else:
        o += ["-", "-"]
    if b is not None:
        for m_ in (O.rare, ~O.rare):
            d = O[m_ & b.notna()]; v, s = clm(d, b[m_ & b.notna()]); o.append(f"{v:+.4f} ({s:.4f})")
    else:
        o += ["-", "-"]
    print(" | ".join(o), flush=True)
print("onset: rare site (G117 fmax <= 0.1, the rarity table without the month for January, February) against the rest")
Gm = pd.read_csv(W + "g119_month_postproc.csv", dtype={"w": str}); M3 = Gm[(Gm.month == "2031-03") & (Gm.cond == "queue_onset")].copy(); Fx = pd.read_csv(W + "g117_onset_oracle.csv").set_index("w").fmax; M3["rare"] = M3.w.map(Fx) <= 0.1
Tr = Gm[(Gm.vis == "mask") & (Gm.cond == "queue_onset")].copy(); Tb = {mo: rarity_lomo(mo) for mo in MONTHS}; fx = []
for r in Tr.itertuples():
    k = int(r.w); lab, el = T.LAB[k]; key = (r.panel, pd.Timestamp(str(T.META["date"][k])[:10]).dayofweek >= 5, (int(T.META["t"][k]) % 288) // 24); Y_ = lab[5] & el[5]; f = Tb[r.month].get(key, np.zeros(len(Y_))); fx.append(float(f[Y_.astype(bool)].max()) if Y_.any() else 1.0)
Tr["rare"] = np.array(fx) <= 0.1; print("windows M3", len(M3), "rare", int(M3.rare.sum()), "| train", len(Tr), "rare", int(Tr.rare.sum()))
print("component | M3 rare | M3 rest | train rare | train rest")
for nm, f_ in (("evidence E-raw", lambda D: D.E - D.raw), ("physics P-raw", lambda D: D.P - D.raw), ("physics on evidence PE-E", lambda D: D.PE - D.E), ("A-raw", lambda D: D.PE - D.raw)):
    o = [nm]
    for D in (M3, Tr):
        for m_ in (D.rare, ~D.rare):
            d = D[m_]; v, s = clm(d, f_(d)); o.append(f"{v:+.4f} ({s:.4f})")
    print(" | ".join(o), flush=True)
