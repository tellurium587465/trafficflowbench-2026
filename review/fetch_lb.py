# Read-only: anonymous public leaderboard view (team, score, submission count, last submission) for TFB.
import json, urllib.request, http.cookiejar, sys
cj = http.cookiejar.CookieJar()
op = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(cj))
op.addheaders = [("User-Agent", "Mozilla/5.0")]
op.open("https://www.kaggle.com/competitions/2026-ieee-big-data-traffic-flow-bench/leaderboard").read()
x = next((c.value for c in cj if "XSRF" in c.name.upper()), "")
req = urllib.request.Request("https://www.kaggle.com/api/i/competitions.LeaderboardService/GetLeaderboard",
                             data=json.dumps({"competitionId": 153956, "leaderboardMode": "LEADERBOARD_MODE_DEFAULT"}).encode(),
                             headers={"content-type": "application/json", "x-xsrf-token": x, "User-Agent": "Mozilla/5.0"})
d = json.loads(op.open(req).read())
rows = d.get("publicLeaderboard") or d.get("teams") or []
json.dump(d, open(r"C:/Users/user/tfb/review/lb_now.json", "w", encoding="utf-8"), ensure_ascii=False)
n = int(sys.argv[1]) if len(sys.argv) > 1 else 40
for r in rows[:n]:
    print(r.get("rank"), r.get("teamId"), r.get("teamName"), r.get("displayScore") or r.get("score"),
          "subs", r.get("submissionCount"), "last", str(r.get("lastSubmissionDate"))[:19], "members", [m.get("userName") for m in r.get("teamMembers", [])])
print("keys:", list(d.keys())[:10], "row keys:", list(rows[0].keys()) if rows else None)

