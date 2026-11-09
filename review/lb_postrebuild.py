# Read-only: classify LB bests as pre/post rebuild using global submission-id ordering, estimate best dates.
import json, numpy as np, datetime as dt
d = json.load(open(r"C:/Users/user/tfb/review/lb_now.json", encoding="utf-8"))
teams = {t["teamId"]: t for t in d["teams"]}
# anchors: (submission id, UTC time) where the best submission IS the team's last submission (single/last known)
A = [(56572610, "2026-09-26T07:12:11"), (56896571, "2026-10-07T01:30:34"), (56899024, "2026-10-07T03:34:55")]
ids = np.array([a for a, _ in A], float); ts = np.array([dt.datetime.fromisoformat(t).timestamp() for _, t in A])
k, b = np.polyfit(ids, ts, 1)
REBUILD_UPPER_BOUND_ID = 56141722   # Kushal's best; his last submission was 2026-09-10 09:35 UTC (< rebuild 09-11 16:25)
rows = []
for r in d["publicLeaderboard"]:
    t = teams.get(r["teamId"], {"teamName": "?", "submissionCount": 0, "lastSubmissionDate": ""}); sid = r.get("submissionId") or 0
    est = dt.datetime.fromtimestamp(k * sid + b, dt.UTC).strftime("%m-%d %H:%M") if sid > 56400000 else "pre-9/11?"
    rows.append((r.get("rank"), t["teamName"], float(r.get("displayScore") or 0), sid, t["submissionCount"], str(t["lastSubmissionDate"])[:16], est))
print("rate: %.0f ids/day" % (86400 / k))
pre = [x for x in rows if x[3] and x[3] <= REBUILD_UPPER_BOUND_ID]
print("bests with id <= Kushal's pre-rebuild best (certainly pre-rebuild):", len(pre))
for x in pre[:12]: print("  PRE", x)
post = [x for x in rows if x[3] > 56400000]
print("post-rebuild ranking (top 15):")
for i, x in enumerate(post[:15], 1): print(f"  {i:2d}", x)
me = [i for i, x in enumerate(post, 1) if x[1] == "tellurium_rrr"]
print("our post-rebuild rank:", me, "of", len(post))
amb = [x for x in rows if REBUILD_UPPER_BOUND_ID < x[3] <= 56400000]
print("ambiguous ids:", amb[:10])

