"""TE-A9: are queue ends pinned at connector junctions (hidden OD on/off flows)?
Per panel (train): congested segments (>= 2 links). Tail = min index (upstream end), head = max index (downstream end).
Indicator conn_in[i]  = link i has a CONN-*-IN among incoming links (a merge of a hidden on-flow)
          conn_out[i] = link i has a CONN-*-OUT among outgoing links (a diverge)
Compares the share of tails at conn_in links (and +-0/1) to the base rate of conn_in links; same for heads/conn_out.
Also: how long a tail stays put (median run length in steps of an unchanged tail index)."""
import sys, numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, r"C:/Users/user/tfb/src")
from cachelib import load, T2_PANELS
from te_a1_bottleneck import segments

rows = []
for p in T2_PANELS:
    P = load(p, "train")
    V = P.V
    C = np.isfinite(V) & (V <= P.vcut[None, :])
    inc = P.topo.incoming_link_ids.fillna("").astype(str); out = P.topo.outgoing_link_ids.fillna("").astype(str)
    cin = inc.str.contains("CONN-").to_numpy() & inc.str.contains("-IN").to_numpy()
    cout = out.str.contains("CONN-").to_numpy() & out.str.contains("-OUT").to_numpy()
    onr = P.on_attached; offr = P.off_attached
    tails, heads = [], []
    for t in range(0, P.T, 2):
        for a, b in segments(C[t]):
            if b - a >= 1 and a > 0 and b < P.n - 1:
                tails.append(a); heads.append(b)
    tails = np.array(tails); heads = np.array(heads)
    rows.append(dict(panel=p, n_seg=len(tails), base_cin=cin.mean(), tail_at_cin=cin[tails].mean(), base_cout=cout.mean(),
                     head_at_cout=cout[heads].mean(), head_next_cout=cout[np.minimum(heads + 1, P.n - 1)].mean(),
                     base_onramp=onr.mean(), tail_at_onramp=onr[tails].mean(), base_offramp=offr.mean(), head_at_offramp=offr[heads].mean()))
    load.cache_clear()
d = pd.DataFrame(rows)
pd.set_option("display.width", 220)
print(d.round(3).to_string(index=False))
print(d.drop(columns=["panel"]).mean().round(3))
