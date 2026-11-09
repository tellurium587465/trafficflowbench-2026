"""B27b (10/29, ledger #69 part 2, strategy J7): the S_FD levels behind S_LWR_perfect = (3 x 0.9636 - S_FD_truth) / 2.
  truth : S_FD of the observation table itself (current network_parameters = fd_parameters authoritative), per training
          month, all 10 panels (proxy formula; equal to the official to 6 digits on February in B27)
  old   : the same with the pre-fix parameters the scorer docstring describes (lanes / free_speed / capacity from links.csv,
          critical_density and k_jam from fd_parameters) - where the "honest 0.8833" of SCORING_SPEC could come from
  March : the official derive_physics + score_panel S_FD of the production T1 state (finA5) on the validation split, real
          templates; non-target cells = the masked layer (the T2 blank cells the organizer fills with truth are absent)
usage: py b27b_t3_sfd_levels.py"""
import sys, json, numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, r"C:/Users/user/tfb/src")
sys.path.insert(0, r"C:/Users/user/tfb/meta/repo_full/src/task3")
import score_task3 as st
from cachelib import load
from common import PANELS, WORK, REL

MODES = json.loads(open(r"C:/Users/user/tfb/meta/repo_full/config/task3_lwr_modes.json", encoding="utf-8").read())


def sfd(Q, V, lanes, vf, cap, kc, kj, ok):
    k = Q / np.maximum(V, 1.0); ql, kl = Q / lanes, k / lanes; capl, kcl, kjl = cap / lanes, kc / lanes, kj / lanes
    qfd = np.where(kl <= kcl, vf * kl, capl / np.maximum(kjl - kcl, 1e-9) * np.maximum(kjl - kl, 0))
    return np.abs(ql - qfd)[ok].sum(), np.abs(ql)[ok].sum()


rows = []
for p in PANELS:
    P = load(p, "train"); nd = REL / "corridors" / p / "network"
    L = pd.read_csv(nd / "links.csv", dtype={"link_id": str}).drop_duplicates("link_id").set_index("link_id").reindex(P.links)
    lanes_o = pd.to_numeric(L.lanes, errors="coerce").fillna(1.0).clip(lower=1.0).to_numpy(float)[None, :]
    vf_o = pd.to_numeric(L.free_speed_kmh, errors="coerce").fillna(100.0).to_numpy(float)[None, :]
    cap_o = pd.to_numeric(L.capacity_vph, errors="coerce").fillna(pd.Series(1800.0 * lanes_o[0], index=L.index)).to_numpy(float)[None, :]
    fd = pd.read_csv(nd / "fd_parameters.csv", dtype={"link_id": str}).groupby("link_id").mean(numeric_only=True).reindex(P.links)
    kc_o = fd.critical_density.to_numpy(float)[None, :] if "critical_density" in fd else (cap_o / np.maximum(vf_o, 1))
    kj_o = np.maximum(fd.k_jam.fillna(100.0 * fd.lanes).to_numpy(float)[None, :], kc_o * 1.05)
    cur = (P.lanes[None, :], P.vf[None, :], P.cap[None, :], P.kc[None, :], P.kj[None, :])
    old = (lanes_o, vf_o, cap_o, kc_o, kj_o)
    mon = np.array([t[:7] for t in P.times])
    for m in np.unique(mon):
        i = mon == m; V, Q = P.V[i], P.Q[i]; ok = P.E[i] & np.isfinite(V) & np.isfinite(Q)
        a, b = sfd(Q, V, *cur, ok); c, d = sfd(Q, V, *old, ok)
        rows.append((p, m, 1 - a / b, 1 - c / d))
    print(p, "truth S_FD by month (current):", " ".join(f"{r[3 - 1]:.4f}" for r in rows if r[0] == p), "| old params:", " ".join(f"{r[3]:.3f}" for r in rows if r[0] == p), flush=True)
    load.cache_clear()
D = pd.DataFrame(rows, columns=["panel", "month", "sfd_cur", "sfd_old"])
pm = D.groupby("month")[["sfd_cur", "sfd_old"]].mean()
print("\ntruth S_FD, panel mean by month:\n" + pm.round(4).to_string())
for col in ("sfd_cur", "sfd_old"):
    v = D.groupby("panel")[col].mean().mean()
    print(f"{col}: mean {v:.4f} -> S_LWR_perfect = (3 x 0.9636 - S_FD) / 2 = {(3 * 0.9636 - v) / 2:.4f}")

# March: the production T1 state through the official scorer (S_FD only)
S = pd.read_parquet(WORK + "/t1/state_f3ens3c_ad2_gapml_clip_fd_bnd.parquet")
out = []
for p in PANELS:
    nd = REL / "corridors" / p
    f0 = sorted((nd / "validation" / "mainline_states_masked").glob("**/*.parquet"))[0]
    s0 = pd.read_parquet(f0, columns=["link_id", "station_id"]).drop_duplicates("link_id")
    sp = S[S.panel == p].merge(s0.astype(str), on="link_id", how="left"); sp["timestamp"] = pd.to_datetime(sp.timestamp, utc=True)
    ramps = st.load_ramp_flows(nd, "validation")
    d = pd.concat([st.derive_physics(p, REL, "validation", sp, r, None, None, ramps, None)[0] for r in ("R1", "R2", "R3")], ignore_index=True)
    sc = [st.score_panel(p, d, REL, MODES, r) for r in ("R1", "R2", "R3")]
    x = d[d.is_target > 0.5]
    out.append((p, np.mean([s["S_FD"] for s in sc]), len(x)))
    print(p, f"March official S_FD (finA5 T1) {out[-1][1]:.5f}  target rows {len(x)}", flush=True)
print(f"March official S_FD, panel mean: {np.mean([o[1] for o in out]):.5f}")
