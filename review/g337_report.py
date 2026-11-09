"""G337 report (#268): reads work/t2/g337_strata.csv. lowcov (R ch2 share < 0.6) is an M3 masking artefact (official / Jan-Jun R coverage
~0.8, M3 ~0.6) and rare is empty: both left out of the strata and reported as facts only. Strata: day band (1-2 / 3-5 / 6+) x weekend
(+ young/small box for ongoing). Decomposition on the strata both instruments have, weights renormalised there:
total = sum (wA - wB) IoU_B [composition] + sum wA (IoU_A - IoU_B) [within]. Expected IoU at the official March / April composition with
each instrument's per-stratum rates (a stratum an instrument lacks takes the instrument's mean), SE by window bootstrap within strata."""
import numpy as np, pandas as pd
W = "C:/Users/user/tfb/work/t2/"; D = pd.read_csv(W + "g337_strata.csv"); rng = np.random.default_rng(0)
D["band"] = np.where(D.dom <= 2, "d1-2", np.where(D.dom <= 5, "d3-5", "d6+"))
print("R coverage < 0.6 share (artefact check): " + ", ".join(f"{i} {D[D.inst == i].lowcov.mean():.2f}" for i in ("M3", "JF", "JUN", "MAR", "APR")))
for cond, keys in (("queue_ongoing", ["band", "weekend", "ys"]), ("queue_onset", ["band", "weekend"])):
    d = D[D.cond == cond].copy(); d["st"] = d[keys].astype(str).agg("/".join, axis=1)
    print(f"=== {cond}", flush=True)
    for a in keys:
        lv = sorted(d[a].astype(str).unique())
        print(f"  {a}: " + " | ".join(f"{i} " + " ".join(f"{v} {(d[d.inst == i][a].astype(str) == v).mean():.2f}" + (f"({d[(d.inst == i) & (d[a].astype(str) == v)].iou.mean():.3f},n{int(((d.inst == i) & (d[a].astype(str) == v)).sum())})" if i in ("M3", "JF", "JUN") else "") for v in lv) for i in ("M3", "JF", "JUN", "MAR", "APR")), flush=True)
    def comp(A_, B_):
        a, b = d[d.inst == A_], d[d.inst == B_]; ia = a.groupby("st").iou.mean(); ib = b.groupby("st").iou.mean(); sts = sorted(set(ia.index) & set(ib.index))
        wa = a.st.value_counts().reindex(sts); wa = wa / wa.sum(); wb = b.st.value_counts().reindex(sts); wb = wb / wb.sum()
        c_ = float(((wa - wb) * ib[sts]).sum()); w_ = float((wa * (ia[sts] - ib[sts])).sum()); return c_ + w_, c_, w_, float(a.st.isin(sts).mean()), float(b.st.isin(sts).mean())
    for A_, B_ in (("M3", "JF"), ("M3", "JUN"), ("JUN", "JF")):
        t_, c_, w_, ca, cb = comp(A_, B_); print(f"  {A_} - {B_} (common strata, {ca:.2f} / {cb:.2f} of windows): {t_:+.4f} = composition {c_:+.4f} + within {w_:+.4f}", flush=True)
    for i in ("M3", "JF", "JUN"):
        x = d[d.inst == i]; out = []
        for tgt in (i, "MAR", "APR"):
            wv = d[d.inst == tgt].st.value_counts(normalize=True); bs = []
            for b_ in range(300):
                gs = {k_: g_.iou.to_numpy() for k_, g_ in x.groupby("st")}; r = {k_: (v_.mean() if b_ == 0 else rng.choice(v_, len(v_)).mean()) for k_, v_ in gs.items()}
                bs.append(float(sum(w_ * r.get(s_, x.iou.mean()) for s_, w_ in wv.items())))
            out.append(f"at {tgt} composition {bs[0]:.4f} +- {np.std(bs[1:]):.4f}")
        miss = d[d.inst == "APR"].st[~d[d.inst == "APR"].st.isin(x.st.unique())]; out.append(f"April windows in strata {i} lacks: {len(miss)}")
        print(f"  {i}: " + " | ".join(out), flush=True)
    print("  April strata (share): " + ", ".join(f"{k} {v:.3f}" for k, v in d[d.inst == "APR"].st.value_counts(normalize=True).items()), flush=True)
