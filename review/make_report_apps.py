"""Report appendices C and E (11/01): C = the E7 draft's closed-candidate tables with the audit's fixes and the rows after
ledger #106; E = every Kaggle submission from report/subs_all.csv (kaggle competitions submissions --csv --page-size 200).
usage: py make_report_apps.py -> report/R2_appC_en.md, R2_appE_en.md, R2_appE_ja.md"""
import csv, re

REP = "C:/Users/user/OneDrive/Desktop/trafficflowbench/report/"
D = open(REP + "E7_full_draft_en.md", encoding="utf-8").read().split("\n")


def table_after(head):
    i = next(k for k, l in enumerate(D) if l.startswith(head)) + 2; out = []
    while D[i].startswith("|"):
        out.append([c.strip() for c in re.split(r"(?<!\\)\|", D[i].strip())[1:-1]]); i += 1
    return out


FIX = {"the generator's adjustment is ridge-like": "The L2 solution's zero set matters; f* is not a ridge on the public links alone (Section 4.4)",
       "Undecided on both instruments: adding nearby months neither clearly helps nor hurts at this resolution, so the premise behind E4 stays unverified on a real month. Candidate B carries no E4 for this reason":
       "Undecided on both instruments: adding nearby months neither clearly helps nor hurts at this resolution, so the premise behind E4 stays unverified on a real month"}
rows = []
for r in table_after("| Idea | Result | Why it failed"):
    rows.append([FIX.get(c, c) for c in r])
for r in table_after("| Mechanism (ledger) |"):
    if r[0].startswith(("X1 on the training months", "The same networks on March")):
        continue
    res = r[1] + ("" if r[2] == "—" else f"; January / February {r[2]}")
    rows.append([r[0], res, r[3]])
rows += [["Direction of onset misses, smoothing or shifting the probabilities (#111)", "miss on one side in 55 %, 56 %, 50 % of windows (line 70 %); neighbours' observed over predicted 1.01 [0.98, 1.04]", "No direction to shift towards and calibrated neighbours, so nothing to add; closed at the diagnostic"],
         ["Task 1 second residual stage on hidden neighbours' first-stage predictions (#112)", "R² 0.121 congested speed, 0.057 blanks; objective −0.00016, halves −0.00027, 0.00000, 3 of 10 corridors positive", "The FD blend and boundary model already take the explainable part"],
         ["Jam age as a feature of the ongoing LightGBM (#113)", "log-loss improvement −0.00157 (SE 0.00063) on March, −0.00164 (SE 0.00102) on January and February", "Ages beyond an hour are already in the probabilities; two readings of hidden rows agree"],
         ["Exact expected IoU with the scoring mask (#114)", "−0.0093 (SE 0.0060, z/1.25 −1.23, corrected −3.09), January +0.0002, February 0", "The decoder is already optimal given the mask's uncertainty"],
         ["Evidence penalty switched by a precursor flag (#115)", "best of 6 cells −0.0175 (SE 0.0075, z/1.25 −1.86, corrected −3.85), January +0.0024, February −0.0008; no cell is non-negative on all three months, even with an oracle flag", "The months' disagreement on the penalty is not a mix of precursor types; March's gain comes from windows with a precursor. January and February have one no-precursor window each, so the reading has little power there"],
         ["M3 ongoing reweighted to the official windows on history dynamics (#116)", "AUC 0.55; level 0.9037 to 0.9014 (SE of the move 0.0042); far gate +0.0038 to +0.0031, X1 +0.0046 to +0.0043, cone, LightGBM and tfm keep their signs", "History dynamics do not separate the official windows from M3, and no component changes sign"],
         ["Speed inputs put back on the training scale per link (#117b)", "free-flow level moves by more than 1 % on 36 % of links in March and 40 % in April, 0 % in training months; onset −0.0008 (SE 0.0008), ongoing +0.0002 (SE 0.0003); labels change on 0.016 % of cells", "The networks are insensitive to this redraw, and the label threshold does not move"],
         ["Onset components read M3's way on January and February (#120)", "the same decoded sets read on eligible cells, on all finite cells and by M3's selection: B 1.0 +0.0084, +0.0076, +0.0089; SKIP +0.0055, +0.0049, +0.0051", "M3's window choice and visible-cell scoring do not change the order of the components"],
         ["Evidence rule split into place and amount (#122)", "against × 0.25 on March: place only −0.0188 (SE 0.0086), amount only −0.0335, no rule −0.0345", "The gain needs both; amount alone adds almost nothing"],
         ["Bundle of same-day X1, X3a 0.2 and temperature 1.2 (#121)", "+0.0056 (SE 0.0032, z/1.25 +1.41), halves +0.0076, +0.0062; January +0.0040, February +0.0016 without same-day X1", "Below the line, positive on every read"],
         ["Row at T as a late update of ongoing (#123)", "log-loss improvement +0.00015 (SE 0.00018) on March, −0.00002 (SE 0.00008) on January and February", "The probabilities are already calibrated within each state at T"],
         ["Task 4: mixing the L2 solution with a second solution along a line fitted to LB points", "not run", "It would estimate a hidden reference from the leaderboard, which our rules exclude"]]
with open(REP + "R2_appC_en.md", "w", encoding="utf-8") as f:
    f.write("# Appendix C. All closed candidates\n\nValues are on the M3 instrument's development windows unless noted. \"LB\" marks a declared public pair.\n\n| Idea | Result | Why it failed, or what we learned |\n|---|---|---|\n")
    for r in rows:
        f.write("| " + " | ".join(r) + " |\n")
print("appendix C rows", len(rows))

S = sorted(csv.DictReader(open(REP + "subs_all.csv", encoding="utf-8")), key=lambda r: r["date"])


def clip(t, n=110):
    t = " ".join(t.replace("|", "/").split())
    return t if len(t) <= n else t[:n - 1].rstrip() + "…"


body = "".join(f"| {i} | {r['date'][:16]} | {r['fileName']} | {r['publicScore']} | {clip(r['description'])} |\n" for i, r in enumerate(S, 1))
open(REP + "R2_appE_en.md", "w", encoding="utf-8").write(f"# Appendix E. All leaderboard submissions\n\nAll {len(S)} submissions in date order (UTC), with the public score and Kaggle's description, shortened to fit.\n\n| # | Date | File | Public | Description |\n|---|---|---|---|---|\n" + body)
open(REP + "R2_appE_ja.md", "w", encoding="utf-8").write(f"# 付録 E. 全提出\n\n全 {len(S)} 件を日付順(UTC)に示す。公開の点と、Kaggle に登録した説明(長いものは切り詰めた)を載せる。\n\n| # | 日時 | ファイル | 公開 | 説明 |\n|---|---|---|---|---|\n" + body)
print("appendix E rows", len(S), {r["status"] for r in S})
