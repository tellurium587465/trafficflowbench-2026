# Read-only: fetch public discussion topic list (anonymous) for TFB competition.
import json, sys, urllib.request, http.cookiejar
cj = http.cookiejar.CookieJar()
op = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(cj))
op.addheaders = [("User-Agent", "Mozilla/5.0")]
op.open("https://www.kaggle.com/competitions/2026-ieee-big-data-traffic-flow-bench/discussion").read()
x = next((c.value for c in cj if "XSRF" in c.name.upper()), "")
def post(ep, body):
    req = urllib.request.Request("https://www.kaggle.com/api/i/" + ep, data=json.dumps(body).encode(),
                                 headers={"content-type": "application/json", "x-xsrf-token": x, "User-Agent": "Mozilla/5.0"})
    return json.loads(op.open(req).read())
mode = sys.argv[1] if len(sys.argv) > 1 else "list"
if mode == "list":
    d = post("discussions.DiscussionsService/GetTopicListByForumId",
             {"category": "TOPIC_LIST_CATEGORY_ALL", "group": "TOPIC_LIST_GROUP_ALL", "customGroupingIds": [], "author": "TOPIC_LIST_AUTHOR_UNSPECIFIED", "myActivity": "TOPIC_LIST_MY_ACTIVITY_UNSPECIFIED", "recency": "TOPIC_LIST_RECENCY_UNSPECIFIED", "filterCategoryIds": [], "searchQuery": "", "sortBy": "TOPIC_LIST_SORT_BY_RECENT", "page": 1, "forumId": 11991069})
    for t in d.get("topics", []):
        print(t["id"], t.get("postDate", "")[:16], "last", t.get("lastCommentPostDate", "")[:16],
              "v", t.get("votes"), "c", t.get("commentCount"), t["title"][:100])
else:
    d = post("discussions.DiscussionsService/GetForumTopicById", {"forumTopicId": int(mode), "includeComments": True})
    json.dump(d, open(f"C:/Users/user/tfb/review/t{mode}.json", "w"), ensure_ascii=False)
    def walk(m, dep=0):
        a = m.get("author", {}) or {}
        print("  " * dep + f"[{m.get('postDate','')[:16]}] {a.get('displayName','?')}: " + (m.get("rawMarkdown") or m.get("content") or "")[:1500].replace("\n", " "))
        for r in m.get("replies", []) or []:
            walk(r, dep + 1)
    ft = d.get("forumTopic", d)
    fm = ft.get("firstMessage")
    if fm: walk(fm)
    for c in ft.get("comments", []) or d.get("comments", []) or []:
        walk(c, 1)

