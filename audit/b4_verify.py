"""B4: sanity checks of the filled state files: which rows changed vs the input, agreement with the linear version
(t1_gapfill), predicted congestion share inside the validation blanks."""
import numpy as np, pandas as pd
W = "C:/Users/user/tfb/work/t1/"
for base, lin, ml in (("state_f3ens3c", "state_f3ens3c_gap", "state_f3ens3c_gapml"), ("state_f3ens3c_ad", "state_f3ens3c_ad_gap", "state_f3ens3c_ad_gapml")):
    a = pd.read_parquet(W + base + ".parquet"); b = pd.read_parquet(W + lin + ".parquet"); c = pd.read_parquet(W + ml + ".parquet")
    assert len(a) == len(c) and (a.timestamp.astype(str).values == c.timestamp.astype(str).values).all() and (a.link_id.astype(str).values == c.link_id.astype(str).values).all()
    ch_ml = (a.speed_kmh.values != c.speed_kmh.values) | (a.flow_vph.values != c.flow_vph.values)
    same_order = len(a) == len(b) and (a.timestamp.astype(str).values == b.timestamp.astype(str).values).all()
    print(f"{ml}: rows {len(c)}, changed vs {base}: {ch_ml.sum()}; finite {np.isfinite(c[['speed_kmh','flow_vph']].values).all()}; min v {c.speed_kmh.min():.2f}, min q {c.flow_vph.min():.1f}")
    if same_order:
        ch_lin = (a.speed_kmh.values != b.speed_kmh.values) | (a.flow_vph.values != b.flow_vph.values)
        both = ch_ml & ch_lin
        print(f"   linear version changed {ch_lin.sum()} rows; overlap {both.sum()}; mean |v_ml - v_lin| {np.abs(c.speed_kmh.values[both]-b.speed_kmh.values[both]).mean():.2f} km/h, "
              f"mean |q_ml - q_lin| {np.abs(c.flow_vph.values[both]-b.flow_vph.values[both]).mean():.1f} vph")
    else:
        print("   linear file has a different row order/length:", len(b))
    k = c.flow_vph.values / c.speed_kmh.values
    print(f"   filled rows: mean v base {a.speed_kmh.values[ch_ml].mean():.1f} -> ml {c.speed_kmh.values[ch_ml].mean():.1f}; share v<65 km/h base {(a.speed_kmh.values[ch_ml]<65).mean():.3f} -> ml {(c.speed_kmh.values[ch_ml]<65).mean():.3f}")
