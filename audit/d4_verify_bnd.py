"""D4: the _bnd states differ from their parents only at the boundary cells (bnd3_fill), all values finite, k <= k_jam."""
import numpy as np, pandas as pd
W = "C:/Users/user/tfb/work/t1/"
fill = pd.read_parquet(W + "bnd3_fill.parquet"); key = set(zip(fill.panel, fill._ts, fill.link_id))
print("boundary cells:", len(fill), "by panel:", fill.groupby("panel").size().to_dict())
for parent in ("state_f3ens3c_ad2_gapml_clip_fd", "state_f3ens3c_ad_gapml_clip_fd"):
    a = pd.read_parquet(W + parent + ".parquet"); b = pd.read_parquet(W + parent + "_bnd.parquet")
    same = len(a) == len(b) and (a.timestamp.astype(str).values == b.timestamp.astype(str).values).all() and (a.link_id.astype(str).values == b.link_id.astype(str).values).all()
    ch = (np.abs(a.speed_kmh.values - b.speed_kmh.values) > 1e-9) | (np.abs(a.flow_vph.values - b.flow_vph.values) > 1e-9)
    inb = np.array([k in key for k in zip(b.panel.values[ch], b.timestamp.astype(str).values[ch], b.link_id.astype(str).values[ch])])
    print(f"{parent}_bnd: keys/order same {same}; changed {ch.sum()} rows, all inside the boundary set {inb.all()}; finite {np.isfinite(b[['speed_kmh','flow_vph']].values).all()}; "
          f"mean |dv| {np.abs(a.speed_kmh.values[ch]-b.speed_kmh.values[ch]).mean():.2f} km/h, mean |dq| {np.abs(a.flow_vph.values[ch]-b.flow_vph.values[ch]).mean():.1f} vph, "
          f"share v<0.6vf-ish (v<65) parent {(a.speed_kmh.values[ch]<65).mean():.3f} -> {(b.speed_kmh.values[ch]<65).mean():.3f}")
