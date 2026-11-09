"""Build a Task 2 submission = an existing hybrid (onset from one blend, ongoing from another, as run_final.sh) + TTA
(the 8 perturbed-input draws, probs_afdemtta<k>__<tag>.pkl, blended per draw with the same specs, then averaged in logit
space; cells 0 in every draw stay 0) + optionally the cone V3 (t2_cone.py), then the production evidence rule
(t2_onset_evidence.py B=0.25 OGB=0.85) and far gate (t2_far_gate.py D=3 CONTIG=1). T1 / T4 as given.
usage: python t2_build_tta_cone.py <name> <onset spec name|s46> <ongoing spec name|s70> <cone 0|1> <state_parquet> <odme_parquet>
       spec names: s46 = t2_specs.json s37 (the q24f blend), s70 = the R24 + R48 blend of run_final.sh"""
import sys, json, pickle, subprocess, numpy as np, pandas as pd
from cachelib import T2_PANELS
from common import REL

name, son, sog, cone, state, odme = sys.argv[1:7]; cone = int(cone)
W = "C:/Users/user/tfb/work/"; SRC = "C:/Users/user/tfb/src"
SPEC = {"s46": json.load(open(f"{SRC}/t2_specs.json"))["s37"]["spec"],
        "s70": "+".join(f"{f}_{c}_s{s}{w}" for s in range(4) for f in ("q43f", "q46f")
                        for c, w in (("attnglob", ":2"), ("attnwp05", ":2"), ("attn1", ":2"), ("drop1", ""), ("r13", "")))}
cond = {}
for sp in ("validation", "private"):
    for p in T2_PANELS:
        w = pd.read_csv(REL / "task2" / p / sp / "window_index.csv"); cond.update(dict(zip(w.window_id, w.condition)))
lg = lambda x: np.log(np.clip(x, 1e-6, 1 - 1e-6) / (1 - np.clip(x, 1e-6, 1 - 1e-6)))


def blend(spec, k):
    parts = [(x.split(":")[0], float(x.split(":")[1]) if ":" in x else 1.0) for x in spec.split("+")]
    pr = {m: pickle.load(open(W + f"t2/probs_afdemtta{k}__{m}.pkl", "rb")) for m, _ in parts}
    tw = sum(w for _, w in parts)
    return {wid: sum(w * pr[m][wid] for m, w in parts) / tw for wid in cond}


draws = []
for k in range(1, 9):
    a, b = blend(SPEC[son], k), blend(SPEC[sog], k)
    draws.append({wid: (a[wid] if c == "queue_onset" else b[wid]) for wid, c in cond.items()})
out = {}
for wid in cond:
    st = np.stack([d[wid] for d in draws]); out[wid] = np.where((st == 0).all(axis=0), 0.0, 1 / (1 + np.exp(-lg(st).mean(axis=0))))
pickle.dump(out, open(W + f"t2/probs_{name}tta.pkl", "wb")); print("TTA hybrid written:", len(out), "windows", flush=True)
src = f"{name}tta"
if cone:
    subprocess.run([sys.executable, "t2_cone.py", src, f"{name}ttacone"], cwd=SRC, check=True); src = f"{name}ttacone"
subprocess.run([sys.executable, "t2_onset_evidence.py", src, f"{name}on", "0.25", "0.85"], cwd=SRC, check=True)
subprocess.run([sys.executable, "t2_far_gate.py", W + f"t2/queue_{name}on.parquet", W + f"t2/queue_{name}.parquet", "3", "1"], cwd=SRC, check=True)
from submit import assemble
print(assemble(pd.read_parquet(state), pd.read_parquet(W + f"t2/queue_{name}.parquet"), pd.read_parquet(odme), f"{name}_final"))
