"""B13c (10/29, lane 11 check 2): site birth rectangles from the training months (B13 events whose birth row is fully
observed, raw0) -> work/b13_sites.parquet: (panel, head) with n >= 30 and modal-tail share >= 0.6 (criteria fixed before
looking at March). Then, on the M3 v3 DEV onset windows only (holdout untouched), the label runs at T+30: share whose head
is a known site (+-1 link), and among those fully observed (+-1 link margin) the share equal to [modal tail, head]."""
import os, sys, pickle, numpy as np, pandas as pd
os.environ.setdefault("TFB_M3VER", "v3")
sys.path.insert(0, r"C:/Users/user/tfb/src")
from common import WORK
import m3v2
D = pd.read_parquet(WORK + "/b13_events.parquet"); D = D[D.raw0]
S = D.groupby(["panel", "head"]).tail0.agg(n="size", tail=lambda s: s.mode().iloc[0], share=lambda s: (s == s.mode().iloc[0]).mean()).reset_index()
S = S[(S.n >= 30) & (S.share >= 0.6)]; S.to_parquet(WORK + "/b13_sites.parquet", index=False)
print("known sites", len(S), "| per panel", S.groupby("panel").size().to_dict())
SITE = {(r.panel, int(r.head)): int(r.tail) for r in S.itertuples()}
M = pickle.load(open(r"C:/Users/user/tfb/work/t2/m3v3_meta.pkl", "rb"))["meta"]
c = dict(runs=0, site=0, site_obs=0, exact=0, wins=0); og = dict(runs=0, all_left=0, all_gone=0, partial=0)
for m in M:
    if m3v2.is_holdout(np.array([m["day"]]))[0]:
        continue
    y, mk = m["y"].astype(bool), m["m"].astype(bool)               # (n, 6)
    if m["cond"] == "queue_onset":
        c["wins"] += 1; v = y[:, 5]; dd = np.diff(np.r_[0, v.astype(int), 0])
        for a, b in zip(np.flatnonzero(dd == 1), np.flatnonzero(dd == -1) - 1):
            c["runs"] += 1; h = next((hh for hh in (b, b - 1, b + 1) if (m["panel"], hh) in SITE), None)
            if h is None:
                continue
            c["site"] += 1; tl = SITE[(m["panel"], h)]; lo, hi = max(0, min(a, tl) - 1), min(len(v) - 1, max(b, h) + 1)
            if mk[lo:hi + 1, 5].all():
                c["site_obs"] += 1; c["exact"] += int(a == tl and b == h)
print(f"M3 v3 dev onset: windows {c['wins']} | label runs at T+30 {c['runs']} | head at a known site {c['site'] / c['runs']:.3f} | "
      f"fully observed {c['site_obs']} | equal to [modal tail, head] {c['exact'] / max(c['site_obs'], 1):.3f}")
