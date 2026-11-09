"""Build a Task 2 submission from per-condition choices (10/22; generalises t2_build_tta_cone.py):
  onset  : the s46 spec (q24f blend); ongoing : the s70 spec (R24 + R48 blend)
  source per condition : base (saved model) | x1 (t2_x1_official.py, fine-tuned on the month's earlier days) | half (mean)
  tta per condition    : 0 | 1 (8 perturbed-input draws, blended per draw, logit mean; cells 0 in every draw stay 0)
  cone (ongoing)       : 0 | 1 (t2_cone.py, fixed direction)
then the production evidence rule (B=0.25, OGB=0.85) and far gate (D=3, CONTIG=1); T1 / T4 as given. The models, mixes and
their env options (E5, E4 April, TTA / mix splits, LightGBM, physics onset) are in t2_final_spec.py (10/23), shared with the
one-window entry t2_window_entry.py.
usage: python t2_build_variant.py <name> <onset source> <onset tta> <ongoing source> <ongoing tta> <cone> <state_parquet> <odme_parquet>"""
import os, sys, json, pickle, subprocess, numpy as np, pandas as pd
import t2_final_spec as FS
from cachelib import T2_PANELS
from common import REL, WORK

name, son, ton, sog, tog, cone, state, odme = sys.argv[1:9]; ton, tog, cone = int(ton), int(tog), int(cone)
W = WORK + "/"; SRC = os.path.dirname(os.path.abspath(__file__))
CFG = FS.config()                                               # 10/23: env options in t2_final_spec.py (shared with t2_window_entry.py)
cond, split = {}, {}
for sp in ("validation", "private"):
    for p in T2_PANELS:
        w = pd.read_csv(REL / "task2" / p / sp / "window_index.csv"); cond.update(dict(zip(w.window_id, w.condition))); split.update(dict.fromkeys(w.window_id, sp))
_C = {}


def probs(prefix, tag):
    key = (prefix, tag)
    if key not in _C:
        if prefix == "e3stats":                                 # E3' / onset shift: the earlier-day statistics, None without
            import collections, t2_e3_official as E3
            _C[key] = collections.defaultdict(lambda: None, E3._stats())
        else:
            _C[key] = pickle.load(open(W + ("t2/physon_a.pkl" if prefix == "physon_a" else f"t2/probs_{prefix}{tag}.pkl"), "rb"))
    return _C[key]


out = {wid: FS.window_probs(wid, c, split[wid], (son, ton, sog, tog), CFG, lambda pre, tag, wid=wid: probs(pre, tag)[wid]) for wid, c in cond.items()}
print(f"mixes: LightGBM ongoing a = {CFG['la']}, E3' {CFG['e3']}, physics onset a = {CFG['pa']} on {sorted(CFG['mix'])}; TTA on {sorted(CFG['tta'])}", flush=True)
pickle.dump(out, open(W + f"t2/probs_{name}raw.pkl", "wb")); print(f"{name}: onset {son} tta{ton} | ongoing {sog} tta{tog} cone{cone}", flush=True)
src = f"{name}raw"
if cone:
    subprocess.run([sys.executable, "t2_cone.py", src, f"{name}cone"], cwd=SRC, check=True); src = f"{name}cone"
subprocess.run([sys.executable, "t2_onset_evidence.py", src, f"{name}on", "0.25", "0.85"], cwd=SRC, check=True)
subprocess.run([sys.executable, "t2_far_gate.py", W + f"t2/queue_{name}on.parquet", W + f"t2/queue_{name}.parquet", "3", "1"], cwd=SRC, check=True)
if CFG["ograw"]:                                                # 10/30 ledger #72: raw ongoing sets on these splits
    print(f"raw ongoing on {sorted(CFG['ograw'])}: windows changed", FS.ograw(W + f"t2/queue_{name}.parquet", out, cond, split, CFG), flush=True)
from submit import assemble
print(assemble(pd.read_parquet(state), pd.read_parquet(W + f"t2/queue_{name}.parquet"), pd.read_parquet(odme), f"{name}_final"))
