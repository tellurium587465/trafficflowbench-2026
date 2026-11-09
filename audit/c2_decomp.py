"""C2: where does the remaining Task 1 error sit?  (input: audit/sim/<panel>.npz from c1, final = ADAPT+gapml+clip+FD0.75)
Cell kinds (exclusive, in this priority): blank (T2 gap) / next to the blank (12 rows before, 12 after) / congestion boundary
(truth congestion state differs from a +-1 link or +-1 step neighbour) / congested interior (truth v < 0.6 vf) / free flow.
Per kind: share of speed and flow squared error, share that is irreducible noise (0.8% v, 2.3% q white), share of the
physics-proxy residual, and the counterfactual gain in S_state / S_phys / total if that kind's error were zero."""
import sys, glob, os, numpy as np, pandas as pd
sys.path.insert(0, r"C:/Users/user/tfb/src")
from cachelib import FAMILY

SIM = "C:/Users/user/tfb/audit/sim/"
KINDS = ["blank", "next_to_blank", "boundary", "congested", "free"]


def kinds(Z):
    TV = Z["TV"]; vf = Z["vf"]; fin = np.isfinite(TV)
    C = np.where(fin, TV < 0.6 * vf[None, :], False)
    diff = np.zeros_like(C)
    for sh, ax in ((1, 0), (-1, 0), (1, 1), (-1, 1)):
        Cn = np.roll(C, sh, axis=ax); fn = np.roll(fin, sh, axis=ax)
        diff |= fin & fn & (Cn != C)
    k = np.full(C.shape, 4, np.int8)
    k[C] = 3; k[diff] = 2; k[Z["near"]] = 1; k[Z["gap"]] = 0
    return k


def analyse(Z, V, Q):
    TV, TQ, lanes, L = Z["TV"], Z["TQ"], Z["lanes"][None, :], Z["length"][None, :]
    K = kinds(Z); out = []
    for r in (0, 1, 2):
        day = (Z["regime"] == r)[:, None]
        m = Z["target"] & day & Z["E"] & np.isfinite(TV) & np.isfinite(TQ)
        if not m.any():
            continue
        sev = (V - TV) ** 2; seq = ((Q - TQ) / lanes) ** 2
        flv = (0.008 * TV) ** 2; flq = (0.023 * TQ / lanes) ** 2
        N = m.sum(); Sv, Sq = sev[m].sum(), seq[m].sum()
        Lr = 0.54 * np.sqrt(Sv / N) / 25 + 0.46 * np.sqrt(Sq / N) / 600
        # physics proxy residuals
        field = Z["E"] & day & np.isfinite(TV) & np.isfinite(TQ); tgt = Z["target"] & field
        Vx = np.where(tgt, V, TV); Qx = np.where(tgt, Q, TQ)
        ok = field & np.isfinite(Vx) & np.isfinite(Qx)
        N1 = Qx / np.maximum(Vx, 1.0) * L; Nt = TQ / np.maximum(TV, 1.0) * L
        dN, dNt = N1[1:] - N1[:-1], Nt[1:] - Nt[:-1]
        tr = ok[1:] & ok[:-1] & Z["on_v"][:-1] & Z["off_v"][:-1]
        res = np.where(tr, np.abs(dN - dNt), 0.0); den = np.abs(dNt[tr]).sum()
        a, b = tgt[:-1] & tr, tgt[1:] & tr
        wa = np.where(a & b, 0.5, np.where(a, 1.0, 0.0)); wb = np.where(a & b, 0.5, np.where(b, 1.0, 0.0))
        cell_res = np.zeros(TV.shape); cell_res[:-1] += res * wa; cell_res[1:] += res * wb
        for ki, kn in enumerate(KINDS):
            g = m & (K == ki)
            Lg = 0.54 * np.sqrt((Sv - sev[g].sum()) / N) / 25 + 0.46 * np.sqrt((Sq - seq[g].sum()) / N) / 600
            out.append(dict(regime=r, kind=kn, n=int(g.sum()), N=int(N), sev=sev[g].sum(), seq=seq[g].sum(), flv=flv[g].sum(), flq=flq[g].sum(),
                            Sv=Sv, Sq=Sq, dS_state=(Lr - Lg) / 3.0, res=cell_res[g & tgt].sum(), res_tot=res.sum(), den=den,
                            dS_lwr=cell_res[g & tgt].sum() / den / 3.0 if den > 0 else 0.0, Lr=Lr / 3.0,
                            rmse_v=np.sqrt(Sv / N), rmse_q=np.sqrt(Sq / N)))
    return out


if __name__ == "__main__":
    rows = []
    for f in sorted(glob.glob(SIM + "*.npz")):
        p = os.path.basename(f)[:-4]; Z = np.load(f)
        for o in analyse(Z, Z["V"], Z["Q"]):
            rows.append(dict(panel=p, **o))
    d = pd.DataFrame(rows)
    d["dS_phys"] = 2.0 / 3.0 * d.dS_lwr
    pd.set_option("display.width", 250)
    # per panel totals: S_state loss and its pieces
    g = d.groupby(["panel", "kind"]).agg(n=("n", "sum"), sev=("sev", "sum"), seq=("seq", "sum"), flv=("flv", "sum"), flq=("flq", "sum"),
                                         res=("res", "sum"), dS_state=("dS_state", "sum"), dS_phys=("dS_phys", "sum")).reset_index()
    tot = g.groupby("panel")[["n", "sev", "seq", "flv", "flq", "res"]].transform("sum")
    for c in ("n", "sev", "seq", "res"):
        g[c + "_share"] = g[c] / tot[c]
    g["noise_v_share"] = g.flv / g.sev.clip(lower=1e-9); g["noise_q_share"] = g.flq / g.seq.clip(lower=1e-9)
    # aggregate as the leaderboard does: S_state = family mean of panel means; S_phys = panel mean
    pans = sorted(g.panel.unique()); fams = sorted({FAMILY[p] for p in pans})
    def famean(series_by_panel):
        return np.mean([np.mean([series_by_panel.get(p, 0.0) for p in pans if FAMILY[p] == f]) for f in fams])
    summ = []
    for kn in KINDS:
        s = g[g.kind == kn].set_index("panel")
        dS = famean(s.dS_state.to_dict()); dP = float(np.mean([s.dS_phys.get(p, 0.0) for p in pans]))
        summ.append(dict(kind=kn, cell_share=s.n.sum() / g.n.sum(), sev_share=float(np.mean(s.sev_share)), seq_share=float(np.mean(s.seq_share)),
                         noise_v=float((s.flv.sum() / s.sev.sum())), noise_q=float(s.flq.sum() / s.seq.sum()), phys_res_share=float(np.mean(s.res_share)),
                         dS_state_if_zero=dS, dS_phys_if_zero=dP, dTotal_if_zero=0.35 * dS + 0.15 * dP))
    S = pd.DataFrame(summ)
    print("panels:", pans)
    print(S.round(4).to_string(index=False))
    # by regime and by panel (loss and where it is)
    lr = d.groupby(["panel", "regime"]).agg(Lr=("Lr", "first"), rmse_v=("rmse_v", "first"), rmse_q=("rmse_q", "first")).reset_index()
    print("\nS_state loss by regime (mean over panels, x3 = per-regime loss):")
    print(lr.groupby("regime")[["Lr", "rmse_v", "rmse_q"]].mean().round(4).to_string())
    pt = g.pivot_table(index="panel", columns="kind", values="dS_state").reindex(columns=KINDS)
    pt["S_state_loss"] = lr.groupby("panel").Lr.sum()
    pt2 = g.pivot_table(index="panel", columns="kind", values="dS_phys").reindex(columns=KINDS)
    print("\nper panel: S_state loss and the part each kind would give back if its error were zero")
    print(pt.round(4).to_string())
    print("\nper panel: S_phys proxy part each kind would give back")
    print(pt2.round(4).to_string())
    g.to_csv(SIM + "decomp_by_panel_kind.csv", index=False); S.to_csv(SIM + "decomp_summary.csv", index=False)
