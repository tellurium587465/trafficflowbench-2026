"""G169 (11/01, ledger #118): the image network on the released March and April, mixed into the final Task 1 output: (1 - w) final + w net on the Task 1 rows, w_out outside the official T2 blanks (rows t+1 .. t+18 of
each official window), w_in inside, then the speed ceiling and the density clip of t1_clip.py. Only the months named are changed. Assembles a zip from A7's queue and Task 4 by submit.assemble; the zip goes to the LB only
after its pre-declaration in the ledger.
usage: py -B -X utf8 g169_unet_apply.py <w_out> <w_in> <months: 03 or 03,04> <name>"""
import sys, glob, time, numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
sys.dont_write_bytecode = True; sys.path.insert(0, r"C:/Users/user/tfb/src")
WO, WI = float(sys.argv[1]), float(sys.argv[2]); MONTHS = sys.argv[3].split(","); NAME = sys.argv[4]
import torch, torch.nn as nn, torch.nn.functional as Fn
from cachelib import load
from common import PANELS, WORK, REL
from cachelib import T2_PANELS
from submit import assemble

s = open("C:/tfb_data/kaggle_up/k_t1unet/tfb-t1-unet.py", encoding="utf-8").read(); exec(s[s.index("class Res(nn.Module)"):s.index("def load_all()")])
torch.set_num_threads(8); nets = []
for f in sorted(glob.glob("C:/tfb_data/kaggle_out/t1unet/unet_s*.pt")):
    d = torch.load(f, map_location="cpu", weights_only=False); n = Net(14, d["_base"]); n.load_state_dict({k: v.float() for k, v in d.items() if not k.startswith("_")}); n.eval(); nets.append(n)
wkd = lambda dd: float(np.datetime64(str(dd)[:10]).astype("datetime64[D]").astype(int) % 7 in (2, 3))
base = WORK + "/t1/state_f3ens3c_ad2_gapml_clip_fd_bnd.parquet"; st = pd.read_parquet(base); v0 = st.speed_kmh.to_numpy().copy(); q0 = st.flow_vph.to_numpy().copy(); v1, q1 = v0.copy(), q0.copy(); t0 = time.time(); nch = 0
for p in PANELS:
    T = load(p, "train"); cap_v = np.nanmax(np.where(np.isfinite(T.V), T.V, np.nan), axis=0)
    for sp in ("validation", "private"):
        S = load(p, sp); cap_v = np.fmax(cap_v, np.nanmax(np.where(np.isfinite(S.V), S.V, np.nan), axis=0))
    for sp, mo in (("validation", "03"), ("private", "04")):
        if mo not in MONTHS:
            continue
        P = load(p, sp); ti = {str(t): i for i, t in enumerate(P.times)}; li = {l: i for i, l in enumerate(P.links)}; vis = np.isfinite(P.V) & np.isfinite(P.Q); stat = [P.vf / 120, P.lanes / 6, P.cap / 12000, np.minimum(P.length, 3) / 3, P.on_attached.astype(np.float32), P.off_attached.astype(np.float32)]
        VU = np.zeros((P.T, P.n)); QU = VU.copy()
        for i in range(P.T // 288):
            a, b = i * 288, (i + 1) * 288; x, fv, fq = image(p, P.V[a:b] / P.vf[None, :], P.Q[a:b] / P.cap[None, :], vis[a:b], np.arange(288), stat, wkd(P.day[i]))
            with torch.no_grad():
                o = sum(n(pad8(torch.from_numpy(x)[None])[0])[..., :P.n] for n in nets) / len(nets)
            VU[a:b] = (o[0, 0].numpy() + fv) * P.vf; QU[a:b] = (o[0, 1].numpy() + fq) * P.cap
        gap = np.zeros(P.T, bool)
        if p in T2_PANELS:
            for r in pd.read_csv(REL / "task2" / p / sp / "window_index.csv").itertuples():
                t = {str(k)[:16]: i for i, k in enumerate(P.times)}[r.forecast_origin[:16]]; gap[t + 1:t + 19] = True
        m = np.flatnonzero((st.panel.to_numpy() == p) & (st.timestamp.str[5:7].to_numpy() == mo)); r_ = st.timestamp.iloc[m].map(ti).to_numpy(); c_ = st.link_id.iloc[m].map(li).to_numpy(); w = np.where(gap[r_], WI, WO)
        vn = (1 - w) * v0[m] + w * np.clip(VU[r_, c_], 3.0, 1.15 * P.vf[c_]); qn = (1 - w) * q0[m] + w * np.clip(QU[r_, c_], 60.0, None); vn = np.maximum(np.minimum(vn, cap_v[c_]), qn / P.kj[c_]); v1[m] = vn; q1[m] = qn; nch += len(m)
        print(p, mo, len(m), f"gap rows {gap.mean():.3f} | speed change rms {np.sqrt(np.mean((vn - v0[m]) ** 2)):.3f} | {time.time() - t0:.0f}s", flush=True)
    load.cache_clear()
st["speed_kmh"] = v1; st["flow_vph"] = q1; out = WORK + f"/t1/state_{NAME}.parquet"; st.to_parquet(out, index=False)
z = assemble(st, pd.read_parquet(WORK + "/t2/queue_finA7.parquet"), pd.read_parquet(WORK + "/t4/odme_l2.parquet"), NAME); print("rows changed", nch, out, z)
