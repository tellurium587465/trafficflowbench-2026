"""One-command judgement for the decision ledger (INSTRUMENT_V3_SPEC §3, 10/23; minimal version).
Input: a scored frame (csv with w, cond, panel, day, hour and per-window scores, as V.frame writes) and two of its columns
A (current) and B (candidate). Output, for the windows outside the hold-out (m3v2.is_holdout; --holdout: ONLY the hold-out
windows, read once for the final pair, 10/23): the weighted paired
difference per condition and T2 (main = April composition, March composition alongside), SE (cluster bootstrap), z and,
for a model A/B, z / 1.25 (A/A, 10/23), the sign-flip permutation p, P(B > A) over 80-window redraws, the windows changed,
odd / even day halves, the panel heterogeneity (DerSimonian-Laird I^2 of per-panel differences), and the corrected
z = z' - E[max of k N(0,1)] for k compared options. --ledger appends the row to DECISION_LEDGER.md.
usage: python judge.py <frame.csv> <A> <B> --name "<part>" [--kind post|model] [--k 1] [--cond queue_onset|queue_ongoing]
       [--ledger]   (set TFB_M3VER to the frame's window-set version; v3 is the instrument since 10/23 04:20)"""
import sys, os, argparse
ap = argparse.ArgumentParser(); ap.add_argument("frame"); ap.add_argument("a"); ap.add_argument("b"); ap.add_argument("--name", required=True)
ap.add_argument("--kind", default="post", choices=("post", "model")); ap.add_argument("--k", type=int, default=1); ap.add_argument("--cond", default="")
ap.add_argument("--ledger", action="store_true")
ap.add_argument("--holdout", action="store_true")   # 10/23: the hold-out windows ONLY (read once, the final pair)
args = ap.parse_args()
sys.argv = [sys.argv[0], "0"]
import numpy as np, pandas as pd
import m3v2

EMAX = {1: 0.0, 2: 0.57, 3: 0.85, 4: 1.03, 5: 1.16, 6: 1.27, 8: 1.42, 12: 1.63}
emax = lambda k: EMAX.get(k, float(np.sqrt(2 * np.log(k)) - 0.5) if k > 12 else EMAX[max(x for x in EMAX if x <= k)])
D = pd.read_csv(args.frame).dropna(subset=[args.a, args.b])
if args.cond:
    D = D[D.cond == args.cond]
D = D[m3v2.is_holdout(D.day) if args.holdout else ~m3v2.is_holdout(D.day)].reset_index(drop=True)
s = m3v2.summary(D, args.a, args.b, "private"); sm = m3v2.summary(D, args.a, args.b, "validation", n_boot=500, n_draw=500)
z = s["t2_weighted"] / s["se"]; zm = z / 1.25 if args.kind == "model" else z; zc = zm - emax(args.k)
halves = [m3v2.summary(D[D.day % 2 == h], args.a, args.b, "private", n_boot=300, n_draw=100)["t2_weighted"] for h in (1, 0)]
pp = []                                                        # per-panel raw differences for the heterogeneity
for p, g in D.groupby("panel"):
    d = (g[args.b] - g[args.a]).to_numpy()
    if len(d) > 2 and d.std() > 0:
        pp.append((d.mean(), d.var(ddof=1) / len(d)))
I2 = float("nan")
if len(pp) > 2:
    y, v = np.array(pp).T; w = 1 / v; mu = (w * y).sum() / w.sum(); Q = (w * (y - mu) ** 2).sum(); I2 = max(0.0, (Q - (len(y) - 1)) / Q) if Q > 0 else 0.0
print(m3v2.fmt(f"{args.name} [{os.environ.get('TFB_M3VER', 'v2') or 'v2'}, {'HOLD-OUT' if args.holdout else 'dev'}]", s))
print(f"  March composition {sm['t2_weighted']:+.4f} | z {z:+.2f}" + (f" -> /1.25 {zm:+.2f}" if args.kind == "model" else "") +
      f" | k {args.k}: corrected z {zc:+.2f} | halves odd {halves[0]:+.4f} / even {halves[1]:+.4f} | panel I2 {I2:.0%}")
verdict = ("harm" if zm <= -2 else ("adopt (point > 0)" if s["t2_weighted"] > 0 else "do not adopt (point <= 0)"))
print("  rule (point > 0 and no harm <= -2 SE):", verdict)
if args.ledger:
    L = open(r"C:/Users/user/OneDrive/Desktop/trafficflowbench/DECISION_LEDGER.md", encoding="utf-8").read()
    n = 1 + max(int(l.split("|")[1]) for l in L.splitlines() if l.startswith("| ") and l.split("|")[1].strip().isdigit())
    row = (f"| {n} | {args.name} | {args.cond or '両方'} | {args.k} | {s['t2_weighted']:+.4f}(3 月の構成 {sm['t2_weighted']:+.4f})| "
           f"{z:+.1f}" + (f"(÷1.25 → {zm:+.2f})" if args.kind == "model" else "") + f" | {zc:+.2f} | {verdict}(v3、半分割 {halves[0]:+.4f}/{halves[1]:+.4f}、I² {I2:.0%})|\n")
    ls = L.split("\n"); last = max(j for j, l in enumerate(ls) if l.startswith("| ") and l.split("|")[1].strip().isdigit())   # after the table's last row
    ls.insert(last + 1, row.rstrip("\n")); L = "\n".join(ls)
    open(r"C:/Users/user/OneDrive/Desktop/trafficflowbench/DECISION_LEDGER.md", "w", encoding="utf-8", newline="\n").write(L); print("  ledger row", n, "added")
