"""Memory-stress yardstick (M1 audit 2.4): each config = several seeds, each with OOFs for the clean inputs and for
eval-time perturbed inputs (_t: time of day shifted, _r: static link columns rolled, _tr: both; written by t2_nn2.py
with ECF=t+r+tr). Labels are unchanged, so the drop measures how much a model leans on time / link identity.
Prints the seed-mean virtual LB (all / onset / ongoing) under the far rule (FAR=3) and the far-rule gain on ongoing
(ongoing with FAR=3 minus without). The LB anchors: far-rule gain +0.021 on the LB vs -0.044 on the clean month CV.

usage: python t2_memcmp.py <name>=<tag_s3,tag_s4> [<name>=...]
"""
import sys, numpy as np
import t2_eval_oof as E

VAR = ("", "_t", "_r", "_tr")


def vlb(tag, far):
    E.FAR = far
    return np.asarray(E.evaluate(tag)[2])


print(f"{'config':10s} {'variant':6s}  vLB    onset  ongoing | far gain (ongoing)")
for arg in sys.argv[1:]:
    name, tags = arg.split("="); tags = tags.split(",")
    for v in VAR:
        a = np.mean([vlb(t + v, 3) for t in tags], axis=0); og0 = np.mean([vlb(t + v, 0)[2] for t in tags])
        print(f"{name:10s} {v or 'clean':6s}  {a[0]:.4f} {a[1]:.4f} {a[2]:.4f} | {a[2] - og0:+.4f}", flush=True)
