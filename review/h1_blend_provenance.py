"""Harness review H1: can the s17 / s23 T2 blend be rebuilt from files alone (no chat memory)?
Rebuild the blended probabilities from component probs_*.pkl with the *guessed* spec and compare
with the saved probs_<submission>.pkl. Read-only."""
import pickle, time, numpy as np
W = "C:/Users/user/tfb/work/t2/"
CASES = {
    "s17_t1f3ens3c_t2v12nppdem_nnDnp6_t4l2": "v12nppdem:6+" + "+".join(f"q7f_dnp_{k}_s{s}" for k in ("pk", "glob") for s in range(3)),
    "s23_t1GapML_t2v14af_nnAf6_t4l2": "v14af:6+" + "+".join(f"q12f_af_{k}_s{s}" for k in ("pk", "glob") for s in range(3)),
}
t0 = time.time()
for name, spec in CASES.items():
    ref = pickle.load(open(W + f"probs_{name}.pkl", "rb"))
    parts = [(float(w), pickle.load(open(W + f"probs_{t}.pkl", "rb")))
             for t, w in (x.split(":") if ":" in x else (x, "1") for x in spec.split("+"))]
    tw = sum(w for w, _ in parts)
    keys = set(ref); assert all(set(d) == keys for _, d in parts), "window sets differ"
    md = max(float(np.abs(sum(w * d[k] for w, d in parts) / tw - ref[k]).max()) for k in keys)
    print(f"{name}: windows {len(keys)}  guessed spec max|dp| = {md:.2e}  ({'MATCH' if md < 1e-9 else 'NO MATCH'})")
print(f"{time.time()-t0:.1f}s")
