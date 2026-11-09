"""G74 (10/22): E4 replica on M3 v2. e4r = train months + March 1-15 + C24 ongoing synthetic + DOM=1 + SIMW=0.5 + EMA
(the E4 production recipe, attnglob, seeds 0,1) vs e4s_b (train months only, same config) and e4s_a (train months +
March 1-15, the 10/22 sanity arm). M3 windows from March 16 on (0-based day >= 15: days 1-15 are in e4r's / e4s_a's
training), hold-out excluded; both conditions; t2_m3v2_gate.score; model A/B -> z also with the A/A factor 1.25.
usage: python g74_e4_replica.py"""
import sys, pickle
sys.path.insert(0, r"C:/Users/user/tfb/src")
sys.argv = [sys.argv[0], "0"]
import t2_m3_gate as G, t2_m3v2_gate as V, m3v2
ld = lambda t: pickle.load(open(G.W + f"probs_m3__{t}.pkl", "rb"))
arm = lambda tag: (lambda a, b: {w: (a[w] + b[w]) / 2 for w in a})(ld(f"{tag}_s0"), ld(f"{tag}_s1"))
P = {t: arm(t) for t in ("e4s_b", "e4s_a", "e4r")}
WS = [w for w in G.WID if G.META[w]["day"] >= 15]
D = V.frame({t: {w: V.score(P[t][w], G.META[w]) for w in WS} for t in P}).dropna(subset=list(P))
D.to_csv(G.W + "g74_e4_replica.csv", index=False); dev = D[~m3v2.is_holdout(D.day)]
print(f"M3 windows from 3/16: {len(D)}, outside the hold-out: {len(dev)}")
for a, b in (("e4s_b", "e4r"), ("e4s_b", "e4s_a"), ("e4s_a", "e4r")):
    s = m3v2.summary(dev, a, b, "private")
    print(m3v2.fmt(f"{a} -> {b} (dev, 3/16+)", s), f"| z x1.25 {s['t2_weighted'] / (1.25 * s['se']):+.2f}")
