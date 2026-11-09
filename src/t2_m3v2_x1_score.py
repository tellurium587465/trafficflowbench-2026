"""Score X1 arms under M3 v2 (10/22; the inline scoring of m3v2_x1.csv made a script for the placebo V9-X1).
Base = probs_m3__<tag> (seeds averaged), arm = probs_m3v2x1<arm>__<tag> keyed (window_id, k); k as expectation
(uncapped kdist; the rules inside cap the evidence at 3). Writes work/t2/m3v2_x1<arm>.csv and prints base vs arm per
condition, plus arm vs the reference arm (default "" = real X1) when given.
usage: python t2_m3v2_x1_score.py <arm> [tags=e4s_a_s0,e4s_a_s1] [reference arm=""]"""
import sys
ARM = sys.argv[1]; TAGS = (sys.argv[2] if len(sys.argv) > 2 else "e4s_a_s0,e4s_a_s1").split(",")
REF = sys.argv[3] if len(sys.argv) > 3 else ""
sys.argv = ["x", "0"]
import pickle, numpy as np
import t2_m3_gate as G, t2_m3v2_gate as V, m3v2

W = G.W
B = [pickle.load(open(W + f"probs_{G.PRE}__{t}.pkl", "rb")) for t in TAGS]
XP = f"probs_m3{G.VER}v2x1" if G.VER else "probs_m3v2x1"               # 10/23: per window-set version
X = [pickle.load(open(W + f"{XP}{ARM}__{t}.pkl", "rb")) for t in TAGS]
R = [pickle.load(open(W + f"{XP}{REF}__{t}.pkl", "rb")) for t in TAGS] if REF != ARM else None
cols = {c: {} for c in ("base", "arm", "ref")}
for w in G.WID:
    m = G.META[w]; base = sum(b[w] for b in B) / len(B); kd = m3v2.kdist(m["day"])

    def sc(p_, k):
        return G.iou(V.onset_pred(p_, m, "cur", k) if m["cond"] == "queue_onset" else V.ongoing_pred(p_, m, True, k), m)
    cols["base"][w] = sum(p * sc(base, k) for k, p in kd.items())
    cols["arm"][w] = sum(p * sc(sum(x[(w, k)] for x in X) / len(X), k) for k, p in kd.items())
    if R:
        cols["ref"][w] = sum(p * sc(sum(x[(w, k)] for x in R) / len(R), k) for k, p in kd.items())
if not R:
    del cols["ref"]
D = V.frame(cols); D.to_csv(W + f"m3v2_x1{ARM}{G.VER}.csv", index=False)
for c in ("queue_onset", "queue_ongoing"):
    g = D[D.cond == c]
    print(m3v2.fmt(f"{c} base -> x1{ARM}", m3v2.summary(g, "base", "arm", "private")))
    if R:
        print(m3v2.fmt(f"{c} x1{ARM} -> x1{REF}", m3v2.summary(g, "arm", "ref", "private")))
