"""B27 (10/29, ledger #69, strategy J7): does the Task 3 proxy (metrics.physics_score, rhs = the observed dN) measure what
the official score_task3.py measures? score_task3.load_boundary_flux: the organizer flux is "projected onto the released
observations, so a participant who reproduces what was published satisfies the conservation identity exactly", i.e.
rhs == N_obs(t+dt) - N_obs(t) on every transition. Structure checked before (22:15): station = link 1:1 on all panels,
topology links = our link set (observable_by_link drops nothing), mode A everywhere.
(1) The Feb harness of B22 (f3big + gapml_cv, 10 simulated T2 blanks per panel, density clip; variants no_gapml / gapml /
    gapml_fd (a = 0.75 = finA5-like) / fd050 / fd100) through the official derive_physics + score_panel, with only files()
    and published_targets() swapped for February and the harness's target set, and the flux built from the truth frame:
    inflow = dN_obs / dt + outflow - r_on + r_off, outflow = the observed flow. Compared with the proxy per panel.
(2) Official S_FD of the production T1 state (finA5) on March (validation split, real templates; non-target cells are
    the masked layer, so the T2 blank cells the organizer fills with truth are missing here) and S_FD of the truth on the
    whole training split (proxy formula, = official to 6 digits on February, printed) -> S_LWR_perfect = (3 x 0.9636 -
    S_FD_truth) / 2 (BASELINES: perfect answer S_physics 0.9636 on validation).
usage: py b27_t3_official_vs_proxy.py [panels...]"""
import sys, copy, glob, json, numpy as np, pandas as pd, lightgbm as lgb, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, r"C:/Users/user/tfb/src")
sys.path.insert(0, r"C:/Users/user/tfb/meta/repo_full/src/task3")
import score_task3 as st
from cachelib import load
from common import PANELS, WORK, REL
from t1_features3 import grid_features3, rows3, base_of3
from t2_demand import profile
from t2_arrival import _dmin
from t1_gapml import view_train, gap_features, G
from metrics import physics_score

MD = WORK + "/t1/"
mv = lgb.Booster(model_file=MD + "speed_f3big.txt"); mq = lgb.Booster(model_file=MD + "flow_f3big.txt")
gm = {ch: lgb.Booster(model_file=MD + f"gapml_{ch}_cv.txt") for ch in ("speed", "flow")}
Z = np.load(WORK + "/t2/gapml_windows.npz", allow_pickle=True); WM = Z["is_sel"]
MODES = json.loads(open(r"C:/Users/user/tfb/meta/repo_full/config/task3_lwr_modes.json", encoding="utf-8").read())
REGS = ("R1", "R2", "R3")


def feb(P):
    f = np.flatnonzero([s.startswith("2031-02") for s in P.times]); a, b = f[0], f[-1] + 1; Q = copy.copy(P)
    for k in ("times", "V", "Q", "O", "P", "E", "regime", "target", "blank", "R", "RE"):
        setattr(Q, k, getattr(P, k)[a:b].copy())
    Q.T = b - a; Q.day = np.array([t[:10] for t in Q.times[::288]])
    return Q, a


def official(p, split, files, tgt, sub, ramps, flux):
    """(per-regime score_panel rows, concatenated physics frame) with files() / published_targets() swapped."""
    st.files = lambda panel_dir, split_: files
    st.published_targets = lambda panel_dir, panel, split_, regime: tgt.get(regime, set())
    frames = [st.derive_physics(p, REL, split, sub, r, None, None, ramps, flux)[0] for r in REGS]
    d = pd.concat(frames, ignore_index=True)
    return [st.score_panel(p, d, REL, MODES, r) for r in REGS], d


def flux_from_truth(d):
    """inflow such that dt (inflow + r_on - outflow - r_off) == N(next) - N on every 5-minute transition."""
    d = d.sort_values(["link_id", "timestamp"]).copy()
    nxt = d.groupby("link_id").accumulation_N.shift(-1); dt = (d.groupby("link_id").timestamp.shift(-1) - d.timestamp).dt.total_seconds() / 3600
    ok = (dt > 0) & (dt <= 5.1 / 60)
    f = d.loc[ok, ["timestamp", "link_id"]].copy()
    f["outflow_vph"] = d.flow_vph[ok]
    f["inflow_vph"] = (nxt[ok] - d.accumulation_N[ok]) / dt[ok] + d.flow_vph[ok] - d.on_ramp_flow_vph[ok] + d.off_ramp_flow_vph[ok]
    return f


RES = {}; FDT = {}
for p in (sys.argv[1:] or PANELS):
    P0 = load(p, "train"); P, a0 = feb(P0); pi = PANELS.index(p)
    x = np.concatenate([[0.0], np.cumsum(P.length[:-1])]); dm = _dmin(P, x); prof = profile(P0, "2031-02")
    wl = sorted({(int(t) - a0, str(c)) for pp, t, c in zip(Z["panel"][WM], Z["t"][WM], Z["cond"][WM])
                 if str(pp) == p and 30 <= int(t) - a0 < P.T - G - 31})[:10]
    Pg = copy.copy(P); Pg.blank = P.blank.copy(); Pg.target = P.target.copy(); gap = np.zeros((P.T, P.n), bool)
    for t, _ in wl:
        gap[t + 1:t + G + 1] = True; Pg.blank[t - 12:t] = False; Pg.target[t - 12:t] = False
    Pg.blank |= gap & np.isfinite(P.V)
    F, _, _, _ = grid_features3(Pg); bv, bq = base_of3(F); r, c = np.nonzero(Pg.target)
    X, _ = rows3(Pg, F, r, c); X = np.column_stack([np.full(len(r), pi, np.float32), X])
    Vp = np.full((P.T, P.n), np.nan); Qp = Vp.copy()
    Vp[r, c] = np.clip((bv[r, c] + mv.predict(X, num_threads=8)) * P.vf[c], 3.0, 1.15 * P.vf[c])
    Qp[r, c] = np.clip((bq[r, c] + mq.predict(X, num_threads=8)) * P.lanes[c], 60.0, None)
    Vp0, Qp0 = Vp.copy(), Qp.copy()
    for t, cnd in wl:
        V, Q = view_train(Pg, t); Xg, _, gbv, gbq, kk, jj = gap_features(Pg, V, Q, t, cnd, prof, dm, x)
        rr = t + kk; tg = Pg.target[rr, jj] & np.isfinite(gbv) & np.isfinite(gbq)
        Vp[rr[tg], jj[tg]] = np.clip(gbv[tg] + gm["speed"].predict(Xg[tg], num_threads=8), 3.0, 1.15 * P.vf[jj[tg]])
        Qp[rr[tg], jj[tg]] = np.clip((gbq[tg] + gm["flow"].predict(Xg[tg], num_threads=8)) * P.lanes[jj[tg]], 60.0, None)
    def fd(Vx, Qx, a=0.75):
        cong = Vx < 0.6 * P.vf[None, :]; k = np.maximum(P.kj[None, :] - Qx / P.w[None, :], 1e-3)
        return np.where(cong & np.isfinite(Vx), (1 - a) * Vx + a * Qx / k, Vx)
    clip = lambda Vx, Qx: np.maximum(Vx, Qx / P.kj[None, :])
    V_g = clip(Vp, Qp)
    VAR = {"no_gapml": (clip(Vp0, Qp0), Qp0), "gapml": (V_g, Qp), "gapml_fd": (clip(fd(V_g, Qp), Qp), Qp),
           "gapml_fd050": (clip(fd(V_g, Qp, 0.5), Qp), Qp), "gapml_fd100": (clip(fd(V_g, Qp, 1.0), Qp), Qp)}
    val = P.ramp_validity()
    # ---- the official side: February files, the harness's targets, the flux projected onto the observations
    nd = REL / "corridors" / p
    files = [__import__("pathlib").Path(f) for f in sorted(glob.glob(str(nd / "train" / "mainline_states" / "**" / "*.parquet"), recursive=True)) if "2031-02" in f]
    stn = pd.concat([pd.read_parquet(f, columns=["link_id", "station_id"]) for f in files[:1]]).drop_duplicates("link_id")
    stn = dict(zip(stn.link_id.astype(str), stn.station_id.astype(str)))
    ts = pd.to_datetime(pd.Series(P.times), utc=True)
    reg = st.regime_of_dates(p, pd.Series(P.times).str[:10])
    reg = np.asarray(reg).astype(str) if np.asarray(reg).dtype.kind in "OUS" else np.array(REGS)[np.asarray(reg)]
    tgt = {g: set() for g in REGS}
    tsl = list(ts)
    for i, j in zip(r, c):
        tgt[reg[i]].add((tsl[i], stn[P.links[j]], P.links[j]))
    ramps = st.load_ramp_flows(nd, "train"); ramps = ramps[ramps.timestamp.astype(str).str.startswith("2031-02")]
    empty = pd.DataFrame({"panel": pd.Series(dtype=str), "timestamp": pd.Series(dtype="datetime64[ns, UTC]"), "station_id": pd.Series(dtype=str),
                          "link_id": pd.Series(dtype=str), "mask_regime": pd.Series(dtype=str), "speed_kmh": pd.Series(dtype=float), "flow_vph": pd.Series(dtype=float)})
    _, dtru = official(p, "train", files, {}, empty, ramps, None)
    flux = flux_from_truth(dtru)
    sc_t, _ = official(p, "train", files, {}, empty, ramps, flux)
    FDT[p] = np.mean([s["S_FD"] for s in sc_t])
    out = {"truth": (np.mean([s["S_FD"] for s in sc_t]), np.mean([s["S_LWR"] for s in sc_t]), *physics_score(P, P.V, P.Q, validity=val)[:2])}
    for nm, (Vx, Qx) in VAR.items():
        sub = pd.DataFrame({"panel": p, "timestamp": P.times[r], "station_id": [stn[P.links[j]] for j in c], "link_id": P.links[c],
                            "mask_regime": reg[r], "speed_kmh": Vx[r, c], "flow_vph": Qx[r, c]})
        sub["timestamp"] = pd.to_datetime(sub.timestamp, utc=True)
        sc, _ = official(p, "train", files, tgt, sub, ramps, flux)
        prx = physics_score(Pg, Vx, Qx, validity=val)
        out[nm] = (np.mean([s["S_FD"] for s in sc]), np.mean([s["S_LWR"] for s in sc]), prx[0], prx[1])
    RES[p] = out
    print(p, "targets", len(r), " | ".join(f"{k} off FD {v[0]:.5f} LWR {v[1]:.5f} / proxy FD {v[2]:.5f} LWR {v[3]:.5f}" for k, v in out.items()), flush=True)
    load.cache_clear()

K = ("truth", "no_gapml", "gapml", "gapml_fd", "gapml_fd050", "gapml_fd100")
R = {k: np.array([RES[p][k] for p in RES]) for k in K}
print("\n(1) mean over panels: official S_FD, official S_LWR, proxy S_FD, proxy S_LWR | max |official - proxy| S_LWR over panels")
for k in K:
    v = R[k]; print(f"  {k:12s} " + " ".join(f"{x:.6f}" for x in v.mean(0)) + f" | {np.abs(v[:, 1] - v[:, 3]).max():.2e}")
for nm, a, b in (("gapml", "no_gapml", "gapml"), ("FD 0.75", "gapml", "gapml_fd"), ("FD 0.5 vs 0.75", "gapml_fd", "gapml_fd050"), ("FD 1.0 vs 0.75", "gapml_fd", "gapml_fd100")):
    d = (R[b] - R[a]).mean(0); print(f"  pair {nm:15s} dS_LWR official {d[1]:+.6f} proxy {d[3]:+.6f} (diff {d[1] - d[3]:+.1e}) | dS_FD official {d[0]:+.6f} proxy {d[2]:+.6f}")
print("\n(2) S_FD of the truth, official, February, mean over panels:", f"{np.mean(list(FDT.values())):.5f}")
