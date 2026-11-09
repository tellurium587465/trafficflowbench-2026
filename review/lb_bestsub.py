# Read-only: parse lb_now.json (anonymous public LB API) -> best submission id per team (ids are global & time-ordered).
import json, sys
d = json.load(open(r"C:/Users/user/tfb/review/lb_now.json", encoding="utf-8"))
teams = {t["teamId"]: t for t in d["teams"]}
n = int(sys.argv[1]) if len(sys.argv) > 1 else 40
print("example team keys:", list(d["teams"][0].keys()))
for r in d["publicLeaderboard"][:n]:
    t = teams.get(r["teamId"], {})
    print(r["rank"], r["teamId"], t.get("teamName"), r["displayScore"], "bestSubId", r.get("submissionId"),
          "subs", t.get("submissionCount"), "last", str(t.get("lastSubmissionDate"))[:19])
want = sys.argv[2:] if len(sys.argv) > 2 else ["tellurium_rrr", "Saivardhan98", "Anubisyy", "Kushal Khemani", "Kanak Khemani", "minghao mei"]
for r in d["publicLeaderboard"]:
    t = teams.get(r["teamId"], {})
    if t.get("teamName") in want:
        print("REF", r["rank"], t.get("teamName"), r["displayScore"], "bestSubId", r.get("submissionId"), "last", str(t.get("lastSubmissionDate"))[:19])
