"""TE-A4: are some links' speeds clipped at a ceiling below the published vf (noise-free plateau)?
Per link (train): vmax, share of finite cells within 0.1% of vmax, vmax/vf(fd), links.csv free speed.
Also checks the same in validation (masked layer)."""
import sys, numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, r"C:/Users/user/tfb/src")
from cachelib import load
from common import PANELS, REL

rows = []
for p in PANELS:
    links = pd.read_csv(REL / "corridors" / p / "network" / "links.csv", dtype={"link_id": str}).drop_duplicates("link_id").set_index("link_id")
    for sp in ("train", "validation", "private"):
        P = load(p, sp)
        V = P.V
        for i in range(P.n):
            v = V[:, i]; v = v[np.isfinite(v)]
            if v.size < 100:
                continue
            vmax = v.max()
            share = float((v >= vmax * 0.999).mean())
            rows.append(dict(panel=p, split=sp, i=i, link=P.links[i], vf=P.vf[i], vf_links=links.free_speed_kmh.get(P.links[i], np.nan),
                             vmax=vmax, vmax_vf=vmax / P.vf[i], plateau=share, med_vf=np.median(v) / P.vf[i]))
        load.cache_clear()
d = pd.DataFrame(rows)
d.to_csv(r"C:/Users/user/tfb/review/te_a4_speedclip.csv", index=False)
pd.set_option("display.width", 250)
pl = d[d.plateau > 0.05]
print("links with plateau share > 5%:", len(pl), "of", len(d))
print(pl.sort_values(["panel", "i", "split"]).round(4).to_string(index=False))
print(d.groupby("split").plateau.describe())
