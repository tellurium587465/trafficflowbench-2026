"""English and Japanese reports must carry the same numbers (11/01). numbers.csv covers only the placeholders, so this compares
the multiset of numeric tokens per section after includes. Headings, code spans and table captions' roman numerals are ignored.
usage: py check_report_numbers.py -> prints the differences by section; exit 1 if any"""
import collections, csv, re, sys

REP = "C:/Users/user/OneDrive/Desktop/trafficflowbench/report/"


def load(name):
    md = open(REP + name, encoding="utf-8").read()
    for _ in range(2):
        md = re.sub(r"\{\{include:([^}]+)\}\}", lambda m: open(REP + m.group(1), encoding="utf-8").read(), md)
    return re.sub(r"\{\{\w+\}\}", "", md)


NUM = re.compile(r"[+−-]?\d+(?:[.,]\d+)*(?:·10[⁻⁰-⁹]+)?")


def tokens(text):
    text = re.sub(r"`[^`]*`", " ", text)                     # code, file names
    text = re.sub(r"(?<![\d.])\d+\s?月", " ", text).replace("Task-", "Task ")   # Japanese month names (English writes them out)
    text = re.sub(r"[A-Za-z_]+\d[\w.]*", " ", text)          # names such as T1, q24f, M3, D7_I10_E
    text = re.sub(r"\(\w[^()]*?(?:19|20)\d\d[a-z]?\)", " ", text)  # author-year
    return collections.Counter(NUM.findall(text))


def sections(md):
    parts = re.split(r"(?m)^(?=# )", md)
    return [p for p in parts if p.strip()]


en, ja = sections(load("R2_en.md")), sections(load("R2_ja.md"))
print("sections", len(en), len(ja))
bad = 0
for e, j in zip(en, ja):
    a, b = tokens(e.split("\n", 1)[-1]), tokens(j.split("\n", 1)[-1])
    d1, d2 = a - b, b - a
    if d1 or d2:
        bad += 1
        print("##", e.split("\n", 1)[0][:60], "| only EN:", dict(d1), "| only JA:", dict(d2))
print("sections differing:", bad)
sys.exit(1 if bad or len(en) != len(ja) else 0)
