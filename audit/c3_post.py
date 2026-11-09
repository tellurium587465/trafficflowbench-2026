"""C3: cheap post-processing candidates on the simulated months (audit/sim/*.npz from c1), scored with the exact local
scorer + physics proxy and aggregated like the leaderboard (S_state family mean, S_phys panel mean).
Base = ADAPT + gapml + clip + FD 0.75 (the current best Task 1)."""
import sys, glob, os, types, numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, r"C:/Users/user/tfb/src")
from cachelib import FAMILY
from metrics import state_score, physics_score

SIM = "C:/Users/user/tfb/audit/sim/"


def P_of(Z):
    return types.SimpleNamespace(target=Z["target"], regime=Z["regime"], E=Z["E"], V=Z["TV"], Q=Z["TQ"], lanes=Z["lanes"], cap=Z["cap"], kc=Z["kc"],
                                 kj=Z["kj"], vf=Z["vf"], length=Z["length"], w=Z["w"])


def dclip(P, V, Q):
    return np.maximum(V, Q / P.kj[None, :]), Q


def fd_v(P, V, Q, a, thr=0.6):
    cong = V < thr * P.vf[None, :]; k = np.maximum(P.kj[None, :] - Q / P.w[None, :], 1e-3)
    return dclip(P, np.where(cong, (1 - a) * V + a * Q / k, V), Q)


def fd_q(P, V, Q, a, thr=0.6):
    cong = V < thr * P.vf[None, :]; qfd = P.w[None, :] * P.kj[None, :] * V / (V + P.w[None, :])
    return dclip(P, V, np.where(cong, (1 - a) * Q + a * qfd, Q))


def released(Z, V):
    """what the evaluator sees around a cell: prediction at targets, released value elsewhere (NaN at other blanks)"""
    return np.where(Z["target"], V, np.where(Z["blank"], np.nan, Z["TV"]))


def tsmooth(Z, P, V, Q, s, what="vq"):
    Vr, Qr = released(Z, V), np.where(Z["target"], Q, np.where(Z["blank"], np.nan, Z["TQ"]))
    def nbmean(M):
        up = np.vstack([M[1:], np.full((1, M.shape[1]), np.nan)]); dn = np.vstack([np.full((1, M.shape[1]), np.nan), M[:-1]])
        return np.nanmean(np.stack([up, dn]), axis=0)
    tg = Z["target"]
    if what == "vq":
        Vn = np.where(tg & np.isfinite(nbmean(Vr)), (1 - s) * V + s * nbmean(Vr), V)
        Qn = np.where(tg & np.isfinite(nbmean(Qr)), (1 - s) * Q + s * nbmean(Qr), Q)
        return dclip(P, Vn, Qn)
    K = Qr / np.maximum(Vr, 1.0); km = nbmean(K); k = Q / np.maximum(V, 1.0)
    ks = np.where(tg & np.isfinite(km), (1 - s) * k + s * km, k)
    return dclip(P, np.where(tg, Q / np.maximum(ks, 1e-3), V), Q)        # keep q, move v to the smoothed density


def score(Z, P, V, Q):
    val = (Z["on_v"], Z["off_v"], None, None)
    s = state_score(P, V, Q); ph = physics_score(P, V, Q, validity=val)
    return s, ph[2]


VARIANTS = {
    "base(FD0.75)": lambda Z, P: (Z["V"], Z["Q"]),
    "noFD": lambda Z, P: (Z["Vc"], Z["Qc"]),
    "FD0.5": lambda Z, P: fd_v(P, Z["Vc"], Z["Qc"], 0.5),
    "FD0.6": lambda Z, P: fd_v(P, Z["Vc"], Z["Qc"], 0.6),
    "FD0.9": lambda Z, P: fd_v(P, Z["Vc"], Z["Qc"], 0.9),
    "FD0.75_thr0.5": lambda Z, P: fd_v(P, Z["Vc"], Z["Qc"], 0.75, 0.5),
    "FD0.75_thr0.7": lambda Z, P: fd_v(P, Z["Vc"], Z["Qc"], 0.75, 0.7),
    "FD0.75_thr0.8": lambda Z, P: fd_v(P, Z["Vc"], Z["Qc"], 0.75, 0.8),
    "FDq0.5(keep v)": lambda Z, P: fd_q(P, Z["Vc"], Z["Qc"], 0.5),
    "FD0.75+FDq0.25": lambda Z, P: fd_q(P, *fd_v(P, Z["Vc"], Z["Qc"], 0.75), 0.25),
    "tsmooth_vq0.15": lambda Z, P: tsmooth(Z, P, Z["V"], Z["Q"], 0.15),
    "tsmooth_vq0.3": lambda Z, P: tsmooth(Z, P, Z["V"], Z["Q"], 0.3),
    "tsmooth_k0.15": lambda Z, P: tsmooth(Z, P, Z["V"], Z["Q"], 0.15, "k"),
    "tsmooth_k0.3": lambda Z, P: tsmooth(Z, P, Z["V"], Z["Q"], 0.3, "k"),
}

if __name__ == "__main__":
    only = sys.argv[1:]
    rows = []
    for f in sorted(glob.glob(SIM + "*.npz")):
        p = os.path.basename(f)[:-4]; Z = dict(np.load(f)); P = P_of(Z)
        for nm, fn in VARIANTS.items():
            if only and nm not in only and nm != "base(FD0.75)":
                continue
            V, Q = fn(Z, P); s, ph = score(Z, P, V, Q)
            rows.append(dict(panel=p, variant=nm, S_state=s, S_phys=ph))
        print(p, "scored", flush=True)
    d = pd.DataFrame(rows); pans = sorted(d.panel.unique()); fams = sorted({FAMILY[p] for p in pans})
    b = d[d.variant == "base(FD0.75)"].set_index("panel")
    out = []
    for nm, g in d.groupby("variant", sort=False):
        g = g.set_index("panel")
        dS = np.mean([np.mean([g.S_state[p] - b.S_state[p] for p in pans if FAMILY[p] == f]) for f in fams])
        dP = np.mean([g.S_phys[p] - b.S_phys[p] for p in pans])
        out.append(dict(variant=nm, dS_state=dS, dS_phys=dP, dTotal=0.35 * dS + 0.15 * dP, panels_better=int(sum((0.35 * (g.S_state[p] - b.S_state[p]) + 0.15 * (g.S_phys[p] - b.S_phys[p])) > 0 for p in pans))))
    pd.set_option("display.width", 200)
    print(pd.DataFrame(out).round(5).to_string(index=False))
    d.to_csv(SIM + "post_variants.csv", index=False)
