"""C2b: reducible part of the remaining error per cell kind.  Noise floor = the same scorer applied to a prediction that is
the truth with fresh multiplicative white noise (speed 0.8%, flow 2.3%) at every target: an estimator that knew the
noiseless state exactly would score like this.  Gain if a kind were brought down to its floor (exact counterfactual:
that kind's squared error / physics residual replaced by the floor's), aggregated like the LB."""
import sys, glob, os, numpy as np, pandas as pd
sys.path.insert(0, r"C:/Users/user/tfb/src"); sys.path.insert(0, r"C:/Users/user/tfb/audit")
from cachelib import FAMILY
from c2_decomp import kinds, KINDS

SIM = "C:/Users/user/tfb/audit/sim/"


def pieces(Z, V, Q, K):
    TV, TQ, lanes, L = Z["TV"], Z["TQ"], Z["lanes"][None, :], Z["length"][None, :]
    out = {}
    for r in (0, 1, 2):
        day = (Z["regime"] == r)[:, None]
        m = Z["target"] & day & Z["E"] & np.isfinite(TV) & np.isfinite(TQ)
        if not m.any():
            continue
        sev = np.where(m, (V - TV) ** 2, 0); seq = np.where(m, ((Q - TQ) / lanes) ** 2, 0)
        field = Z["E"] & day & np.isfinite(TV) & np.isfinite(TQ); tgt = Z["target"] & field
        Vx = np.where(tgt, V, TV); Qx = np.where(tgt, Q, TQ); ok = field & np.isfinite(Vx) & np.isfinite(Qx)
        N1 = Qx / np.maximum(Vx, 1.0) * L; Nt = TQ / np.maximum(TV, 1.0) * L
        tr = ok[1:] & ok[:-1] & Z["on_v"][:-1] & Z["off_v"][:-1]
        res = np.where(tr, np.abs((N1[1:] - N1[:-1]) - (Nt[1:] - Nt[:-1])), 0.0); den = np.abs((Nt[1:] - Nt[:-1])[tr]).sum()
        a, b = tgt[:-1] & tr, tgt[1:] & tr
        wa = np.where(a & b, 0.5, np.where(a, 1.0, 0.0)); wb = np.where(a & b, 0.5, np.where(b, 1.0, 0.0))
        cr = np.zeros(TV.shape); cr[:-1] += res * wa; cr[1:] += res * wb
        out[r] = dict(N=m.sum(), den=den, sev={k: sev[K == i].sum() for i, k in enumerate(KINDS)}, seq={k: seq[K == i].sum() for i, k in enumerate(KINDS)},
                      res={k: cr[(K == i) & tgt].sum() for i, k in enumerate(KINDS)})
    return out


def loss(pc):
    return 0.54 * np.sqrt(sum(pc["sev"].values()) / pc["N"]) / 25 + 0.46 * np.sqrt(sum(pc["seq"].values()) / pc["N"]) / 600


rows = []
for f in sorted(glob.glob(SIM + "*.npz")):
    p = os.path.basename(f)[:-4]; Z = np.load(f); K = kinds(Z)
    rng = np.random.default_rng(7)
    Vf = Z["TV"] * (1 + 0.008 * rng.standard_normal(Z["TV"].shape)); Qf = Z["TQ"] * (1 + 0.023 * rng.standard_normal(Z["TQ"].shape))
    cur, flo = pieces(Z, Z["V"], Z["Q"], K), pieces(Z, Vf, Qf, K)
    for r in cur:
        c, fl = cur[r], flo[r]
        Lc = loss(c); Lf = loss(fl)
        rows.append(dict(panel=p, regime=r, kind="ALL", dS_state=(Lc - Lf) / 3, dS_lwr=(sum(c["res"].values()) - sum(fl["res"].values())) / c["den"] / 3,
                         S_lwr=1 - sum(c["res"].values()) / c["den"], S_lwr_floor=1 - sum(fl["res"].values()) / c["den"]))
        for k in KINDS:
            mix = dict(N=c["N"], sev=dict(c["sev"]), seq=dict(c["seq"])); mix["sev"][k] = fl["sev"][k]; mix["seq"][k] = fl["seq"][k]
            rows.append(dict(panel=p, regime=r, kind=k, dS_state=(Lc - loss(mix)) / 3, dS_lwr=(c["res"][k] - fl["res"][k]) / c["den"] / 3))
d = pd.DataFrame(rows); d["dS_phys"] = 2 / 3 * d.dS_lwr
g = d.groupby(["panel", "kind"])[["dS_state", "dS_phys"]].sum().reset_index()
pans = sorted(g.panel.unique()); fams = sorted({FAMILY[p] for p in pans})
out = []
for k in ["ALL"] + KINDS:
    s = g[g.kind == k].set_index("panel")
    dS = np.mean([np.mean([s.dS_state[p] for p in pans if FAMILY[p] == f]) for f in fams]); dP = s.dS_phys.mean()
    out.append(dict(kind=k, dS_state_to_floor=dS, dS_phys_to_floor=dP, dTotal_to_floor=0.35 * dS + 0.15 * dP))
pd.set_option("display.width", 200)
print(pd.DataFrame(out).round(5).to_string(index=False))
a = d[d.kind == "ALL"]
print("S_LWR proxy now vs noise floor (mean over panel x regime):", round(a.S_lwr.mean(), 4), round(a.S_lwr_floor.mean(), 4))
print("by regime, gain to floor (dS_state, dS_phys):", a.groupby("regime")[["dS_state", "dS_phys"]].mean().mul(3).round(4).to_dict())
pt = g.pivot_table(index="panel", columns="kind", values="dS_state").reindex(columns=["ALL"] + KINDS)
print("per panel, S_state gain if brought to the floor:\n", pt.round(4).to_string())
