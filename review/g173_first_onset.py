"""G173 (11/01, ledger #119 addendum, review A3): M3 dev onset windows by whether they could be an official window: the official selector takes the first onsets of the month, first come, 6 h apart. Births on the panel: rows with a queue seen and none in the
17 rows before, at least 24 rows apart (G119's month_events rule), read from the same day only, so hold-out days are not read. Layers: first birth of the day or not; a birth in the 6 hours before or not. Final A's IoU (g155), rare-site share, evidence and physics (g119's March rows).
usage: TFB_M3VER=v3 py -B -X utf8 g173_first_onset.py"""
import sys
R = r"C:/Users/user/tfb/review/"; sys.dont_write_bytecode = True
src = open(R + "g119_month_postproc.py", encoding="utf-8").read().split('if __name__ == "__main__":')[0]
sys.argv = [sys.argv[0], "run"]; exec(compile(src, "g119", "exec"), globals())
ON = pd.read_csv(W + "g155_onset_b_m3.csv"); O = pd.read_csv(W + "g117_onset_oracle.csv").set_index("w"); Gm = pd.read_csv(W + "g119_month_postproc.csv", dtype={"w": str}); Gm = Gm[(Gm.month == "2031-03") & (Gm.cond == "queue_onset")].set_index("w")
rows = []
for r in ON.itertuples():
    m = G.META[r.w]; P, _, q = m3v2._panel(m["panel"]); t = m["t"]; d0 = t - t % 288; b = t + 6; anyq = q[d0:b].any(axis=1); births = [x + d0 for x in range(17, len(anyq)) if anyq[x] and not anyq[x - 17:x].any()]; births = [x for i, x in enumerate(births) if i == 0 or x - births[i - 1] >= 24]
    rows.append(dict(w=r.w, panel=r.panel, day=r.day, iou=r.A_u, first=not births, near6=any(b - x < 72 for x in births), rare=bool(O.fmax.get(r.w, 1) <= 0.1), ev=Gm.E[r.w] - Gm.raw[r.w], ph=Gm.PE[r.w] - Gm.E[r.w]))
D = pd.DataFrame(rows); D.to_csv(W + "g173_first.csv", index=False)


def clm(d, v, n_boot=2000):
    ids = pd.factorize(d.panel + "_" + d.day.astype(str))[0]; a = np.asarray(v, float); M = np.bincount(ids, a); N = np.bincount(ids); Rm = np.random.default_rng(0).multinomial(len(M), np.full(len(M), 1 / len(M)), size=n_boot); return a.mean(), float(((Rm @ M) / (Rm @ N)).std())


print("windows", len(D), "first of the day", int(D["first"].sum()), "birth in the 6 h before", int(D.near6.sum()))
for nm, m_ in (("first of the day", D["first"]), ("not first", ~D["first"]), ("no birth in the 6 h before", ~D.near6), ("birth in the 6 h before", D.near6)):
    d = D[m_]; print(f"{nm:28s}: {len(d):3d} windows, IoU {clm(d, d.iou)[0]:.4f} ({clm(d, d.iou)[1]:.4f}), rare {d.rare.mean():.3f}, evidence {clm(d, d.ev)[0]:+.4f} ({clm(d, d.ev)[1]:.4f}), physics {clm(d, d.ph)[0]:+.4f} ({clm(d, d.ph)[1]:.4f})")
