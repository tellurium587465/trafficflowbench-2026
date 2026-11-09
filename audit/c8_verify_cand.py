"""C8: the gapml2 candidates differ from their parents only inside the Task 2 blanks."""
import numpy as np, pandas as pd
W = "C:/Users/user/tfb/work/t1/"
fill = pd.read_parquet(W + "gapml2_fill.parquet"); key = set(zip(fill.panel, fill._ts, fill.link_id))
for parent, cand in (("state_f3ens3c_ad_gapml_clip_fd", "state_f3ens3c_ad_cand_gapml2"), ("state_f3ens3c_ad2_gapml_clip_fd", "state_f3ens3c_ad2_cand_gapml2")):
    a = pd.read_parquet(W + parent + ".parquet"); b = pd.read_parquet(W + cand + ".parquet")
    same = len(a) == len(b) and (a.timestamp.astype(str).values == b.timestamp.astype(str).values).all() and (a.link_id.astype(str).values == b.link_id.astype(str).values).all()
    ch = (np.abs(a.speed_kmh.values - b.speed_kmh.values) > 1e-9) | (np.abs(a.flow_vph.values - b.flow_vph.values) > 1e-9)
    inb = np.array([k in key for k in zip(b.panel.values[ch], b.timestamp.astype(str).values[ch], b.link_id.astype(str).values[ch])])
    print(f"{cand}: same keys/order {same}; changed rows {ch.sum()} of {len(b)}, inside the blanks {inb.sum()}; finite {np.isfinite(b[['speed_kmh','flow_vph']].values).all()}; "
          f"mean |dv| {np.abs(a.speed_kmh.values[ch]-b.speed_kmh.values[ch]).mean():.2f} km/h, mean |dq| {np.abs(a.flow_vph.values[ch]-b.flow_vph.values[ch]).mean():.1f} vph")
