"""G210 (11/02, ledger #175): pseudo labels for the image network's month fine-tune: the chain's Task 1 output without the net (state_f3ens3c_ad2_gapml_clip_fd_bnd) on the Task 1 cells of March and April, as (T, n) speed / vf and flow / cap, NaN elsewhere. The net plays no part in them, so reading even days stays clean of its own fine-tune. Writes C:/tfb_data/kaggle_up/d_t1ps/ps_<panel>.npz.
usage: py -B -X utf8 g210_pseudo_build.py"""
import sys, numpy as np, pandas as pd
sys.dont_write_bytecode = True; sys.path.insert(0, r"C:/Users/user/tfb/src")
from cachelib import load
from common import PANELS, WORK
st = pd.read_parquet(WORK + "/t1/state_f3ens3c_ad2_gapml_clip_fd_bnd.parquet"); O = "C:/tfb_data/kaggle_up/d_t1ps/"; a = st.panel.to_numpy(); mo = st.timestamp.str[5:7].to_numpy()
for p in PANELS:
    out = {}
    for sp, m_ in (("validation", "03"), ("private", "04")):
        P = load(p, sp); ti = {str(t): i for i, t in enumerate(P.times)}; li = {l: i for i, l in enumerate(P.links)}; k = np.flatnonzero((a == p) & (mo == m_)); r = st.timestamp.iloc[k].map(ti).to_numpy(); c = st.link_id.iloc[k].map(li).to_numpy(); V = np.full((P.T, P.n), np.nan, np.float32); Q = V.copy(); V[r, c] = st.speed_kmh.to_numpy()[k] / P.vf[c]; Q[r, c] = st.flow_vph.to_numpy()[k] / P.cap[c]; out[sp + "_PV"] = V; out[sp + "_PQ"] = Q
    np.savez_compressed(O + f"ps_{p}.npz", **out); print(p, flush=True); load.cache_clear()
