"""G363 (11/07, after the private LB, record only): the public / private scores of all 96 submissions (read once, user-approved, from the
Kaggle API on 11/07 -> work/subs/all_submissions_private_20261107.csv). (1) correlation and OLS slope of private on public, all
submissions and those since the release re-issue of 2026-09-12; (2) the private top 15; (3) differences (public and private) along the
pairs and chains the ledger read on the LB. usage: py -B -X utf8 g363_private_lb.py"""
import numpy as np, pandas as pd
D = pd.read_csv("C:/Users/user/tfb/work/subs/all_submissions_private_20261107.csv")
D["pub"] = pd.to_numeric(D.publicScore, errors="coerce"); D["prv"] = pd.to_numeric(D.privateScore, errors="coerce"); D["day"] = D.date.str[:10]
S = D.set_index("candidate")


def fit(d, lab):
    d = d.dropna(subset=["pub", "prv"]); r = float(np.corrcoef(d.pub, d.prv)[0, 1]); b, a = np.polyfit(d.pub, d.prv, 1)
    rs = float(d.pub.rank().corr(d.prv.rank()))
    print(f"{lab}: n {len(d)}, Pearson r {r:+.3f}, Spearman {rs:+.3f}, slope private ~ public {b:+.3f} (intercept {a:+.3f})")


print("=== (1) public vs private")
fit(D, "all submissions")
fit(D[D.day >= "2026-09-12"], "since 2026-09-12")
fit(D[D.candidate.str.startswith("fin")], "fin* candidates only")
fit(D[D.candidate.str.startswith("fin") & ~D.candidate.str.contains("_on0|_og0")], "fin* without the on0 / og0 probes")
print("\n=== (2) private top 15")
T = D.sort_values("prv", ascending=False).head(15)
for i, r in enumerate(T.itertuples(), 1):
    print(f"{i:2d}. {r.candidate:24s} private {r.prv:.5f} | public {r.pub:.5f} | {r.day} | {r.ledger_refs if isinstance(r.ledger_refs, str) else ''}")
print("\n=== (3) pairs / chains (second minus first; public | private)")
chains = [["finA21", "finA25"], ["finB7n", "finB7c"], ["finA25", "finA22c"], ["finA21", "finA22c"],
          ["finA13", "finA15", "finA16", "finA17", "finA21"], ["finA7", "finA8", "finA12", "finA13"],
          ["finA7", "finA8", "finA8n", "finA10", "finA11", "finA12", "finA13"], ["finB7", "finB10", "finB11"],
          ["finB7", "finB9", "finB9sn", "finB10", "finB11", "finB12"], ["finB7", "finB7n"], ["e4p0", "e4p1"],
          ["finA17", "finA21"], ["finA13", "finA13_on0"], ["finA13", "finA13_og0"], ["finA6", "finA6_on0"], ["finA6", "finA6_og0"],
          ["finA5", "t1fd050_finA5"], ["finA5", "t1fd100_finA5"], ["finA5", "t1f3all3_finA5"], ["finA7", "finA7unet"], ["finA7", "finA7unet2"],
          ["finA7", "finA7unet3"], ["finA8", "finA8skip"], ["finB11", "finB12"], ["finA21", "probe270a"], ["finA21", "probe270b"],
          ["finA4", "finA5"], ["finA5", "finA6"], ["finA6", "finA7"], ["finB4", "finB5"], ["finB5", "finB6"], ["finB6", "finB7"], ["s76", "finA4"]]
rows = []
for ch in chains:
    if not all(c in S.index for c in ch):
        print("  (missing)", " -> ".join(ch)); continue
    steps = " ; ".join(f"{a}->{b}: {S.pub[b] - S.pub[a]:+.5f} | {S.prv[b] - S.prv[a]:+.5f}" for a, b in zip(ch, ch[1:]))
    tot = f"total {S.pub[ch[-1]] - S.pub[ch[0]]:+.5f} | {S.prv[ch[-1]] - S.prv[ch[0]]:+.5f}" if len(ch) > 2 else ""
    print("  " + steps + ("  || " + tot if tot else ""))
    for a, b in zip(ch, ch[1:]):
        rows.append(dict(first=a, second=b, d_pub=S.pub[b] - S.pub[a], d_prv=S.prv[b] - S.prv[a]))
P = pd.DataFrame(rows).drop_duplicates(["first", "second"]); P = P[~P.second.str.startswith("probe")]
same = (np.sign(P.d_pub) == np.sign(P.d_prv)) | ((P.d_pub.abs() < 5e-6) & (P.d_prv.abs() < 5e-6))
print(f"\nsign agreement public vs private over the {len(P)} distinct steps (probes left out): {int(same.sum())} / {len(P)}; "
      f"r of the step differences {float(np.corrcoef(P.d_pub, P.d_prv)[0, 1]):+.3f}")
P.to_csv("C:/Users/user/tfb/work/subs/g363_pair_steps.csv", index=False)
