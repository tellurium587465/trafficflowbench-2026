"""Task 1, last stage (ledgers #118, #118b, #132): the image network mixed into the chain's output. (1 - w) chain + w net on the Task 1 rows, w_out outside the official T2 blanks (rows t+1 .. t+18 of each official
window), w_in inside, then the speed ceiling and the density clip of t1_clip.py. The net's predictions are the month-fine-tuned (T, n) speed and flow saved by t1_unet_ft.py (run on a GPU, shipped in the assets),
ft_s<seed>_<03|04>_e<epochs>.npz, mean of the seeds found, so this stage needs no torch.
usage: py -B -X utf8 t1_unet_apply.py <w_out> <w_in> <in.parquet> <out.parquet> [epochs, default 32]"""
import os, sys, glob, re, numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
from cachelib import load, T2_PANELS
from common import PANELS, WORK, REL

WO, WI = float(sys.argv[1]), float(sys.argv[2]); FIN, FOUT = sys.argv[3], sys.argv[4]; EP = (sys.argv[5:] or ["32"])[0]; D = os.environ.get("TFB_T1_UNET_DIR", WORK + "/t1/unet_ft/")
SEEDS = {mo: sorted(int(re.search(r"ft_s(\d+)_", f).group(1)) for f in glob.glob(D + f"ft_s*_{mo}_e{EP}.npz")) for mo in ("03", "04")}; assert SEEDS["03"] and SEEDS["03"] == SEEDS["04"], SEEDS
Z = {mo: [dict(np.load(D + f"ft_s{s}_{mo}_e{EP}.npz")) for s in SEEDS[mo]] for mo in SEEDS}
st = pd.read_parquet(FIN); v0 = st.speed_kmh.to_numpy().copy(); q0 = st.flow_vph.to_numpy().copy(); v1, q1 = v0.copy(), q0.copy()
for p in PANELS:
    T = load(p, "train"); cap_v = np.nanmax(np.where(np.isfinite(T.V), T.V, np.nan), axis=0)
    for sp in ("validation", "private"):
        S = load(p, sp); cap_v = np.fmax(cap_v, np.nanmax(np.where(np.isfinite(S.V), S.V, np.nan), axis=0))
    for sp, mo in (("validation", "03"), ("private", "04")):
        P = load(p, sp); ti = {str(t): i for i, t in enumerate(P.times)}; li = {l: i for i, l in enumerate(P.links)}; n_ = len(Z[mo])
        VU = sum(z[p + "_V"].astype(float) for z in Z[mo]) / n_; QU = sum(z[p + "_Q"].astype(float) for z in Z[mo]) / n_; gap = np.zeros(P.T, bool)
        if p in T2_PANELS:
            for r in pd.read_csv(REL / "task2" / p / sp / "window_index.csv").itertuples():
                t = {str(k)[:16]: i for i, k in enumerate(P.times)}[r.forecast_origin[:16]]; gap[t + 1:t + 19] = True
        m = np.flatnonzero((st.panel.to_numpy() == p) & (st.timestamp.str[5:7].to_numpy() == mo)); r_ = st.timestamp.iloc[m].map(ti).to_numpy(); c_ = st.link_id.iloc[m].map(li).to_numpy(); w = np.where(gap[r_], WI, WO)
        vn = (1 - w) * v0[m] + w * np.clip(VU[r_, c_], 3.0, 1.15 * P.vf[c_]); qn = (1 - w) * q0[m] + w * np.clip(QU[r_, c_], 60.0, None); v1[m] = np.maximum(np.minimum(vn, cap_v[c_]), qn / P.kj[c_]); q1[m] = qn
    load.cache_clear()
st["speed_kmh"] = v1; st["flow_vph"] = q1; st.to_parquet(FOUT, index=False); print("seeds", SEEDS["03"], "epochs", EP, "w", WO, WI, "rows", len(st), FOUT)
