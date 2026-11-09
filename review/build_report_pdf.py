"""Report PDF (10/24): the E7 Markdown draft -> one HTML page in an IEEE-like two-column layout -> PDF by headless
Microsoft Edge (no TeX / pandoc on this machine, no package download). Handles what the draft uses: the cover bullet
block, headings (section numbers decide the level), paragraphs, nested bullet lists, pipe tables (escaped \\|), **bold**,
*italic*, `code`. Figures are not embedded in the draft: each Fig. N goes after the first block that cites it, with its
caption from figs/CAPTIONS.md (PNG at 300 dpi; Figs. 6, 8 and tables with > 4 columns span both columns).
usage: python build_report_pdf.py [draft.md=E7_full_draft_en.md] [out.pdf=E7_report_en.pdf]   (paths in the report folder)"""
import os, re, sys, html, glob, subprocess, csv
from pypdf import PdfReader

REP = "C:/Users/user/OneDrive/Desktop/trafficflowbench/report/"
FDIR = sys.argv[3] if len(sys.argv) > 3 else "figs"          # 11/01: the R2 report uses figs2 and numbers.csv
WIDE = {2} if FDIR == "figs2" else {6, 8}
SRC = REP + (sys.argv[1] if len(sys.argv) > 1 else "E7_full_draft_en.md"); OUT = REP + (sys.argv[2] if len(sys.argv) > 2 else "E7_report_en.pdf")
JA = "_ja" in SRC
EDGE = r"C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe"
CAP = {int(m.group(1)): m.group(2).strip() for m in re.finditer(r"^- \*\*Fig\. (\d+)\*\* (.+)$", open(REP + FDIR + ("/CAPTIONS_ja.md" if JA and FDIR == "figs2" else "/CAPTIONS.md"), encoding="utf-8").read(), re.M)}
FIG = {int(re.search(r"fig(\d+)_", os.path.basename(f)).group(1)): os.path.basename(f) for f in glob.glob(REP + FDIR + "/fig*_*.png")}


def inline(s):
    s = html.escape(s.replace("\\|", "\x00"), quote=False).replace("\x00", "|")
    s = re.sub(r"`([^`]+)`", r"<code>\1</code>", s)
    s = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", s)
    s = re.sub(r"\*(?![\s*])(.+?)(?<![\s*])\*" if JA else r"(?<![\w*])\*(?!\s)(.+?)(?<!\s)\*(?![\w*])", r"<i>\1</i>", s)   # ja: kana are \w
    return re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", s)


def cells(row):
    return [c.strip() for c in re.split(r"(?<!\\)\|", row.strip())[1:-1]]


def table(rows):
    head, body = cells(rows[0]), [cells(r) for r in rows[2:]]
    wide = len(head) > 4
    h = "".join(f"<th>{inline(c)}</th>" for c in head)
    b = "".join("<tr>" + "".join(f"<td>{inline(c)}</td>" for c in r) + "</tr>" for r in body)
    return f'<table class="{"wide" if wide else ""}"><thead><tr>{h}</tr></thead><tbody>{b}</tbody></table>'


def heading(t):
    t = t.lstrip("#").strip()
    lvl = 2 if re.match(r"^(\d+\.|Summary|Abstract|References|Appendix|要約|参考文献|付録)", t) and not re.match(r"^\d+\.\d", t) else (3 if re.match(r"^\d+\.\d+\s", t) else 4)
    return f"<h{lvl}>{inline(t)}</h{lvl}>"


def lists(lines):
    out, depth = [], -1
    for l in lines:
        d = (len(l) - len(l.lstrip(" "))) // 2; txt = l.strip()[2:]
        while depth < d:
            out.append("<ul>"); depth += 1
        while depth > d:
            out.append("</li></ul>"); depth -= 1
        if out[-1] != "<ul>":
            out.append("</li>")
        out.append(f"<li>{inline(txt)}")
    out.append("</li>" + "</ul>" * (depth + 1))
    return "".join(out)


def blocks(md):
    lines = md.splitlines(); i = 0
    while i < len(lines):
        l = lines[i]
        if not l.strip():
            i += 1; continue
        if l.startswith("#"):
            yield "h", heading(l); i += 1; continue
        if l.startswith("|"):
            j = i
            while j < len(lines) and lines[j].startswith("|"):
                j += 1
            yield "t", table(lines[i:j]); i = j; continue
        if re.match(r"\d+\. ", l):
            j = i; acc = []
            while j < len(lines) and re.match(r"\d+\. ", lines[j]):
                acc.append(lines[j].split(". ", 1)[1]); j += 1
            yield "l", "<ol>" + "".join(f"<li>{inline(a)}</li>" for a in acc) + "</ol>"; i = j; continue
        if l.lstrip().startswith("- "):
            j = i; acc = []
            while j < len(lines) and lines[j].strip() and (lines[j].lstrip().startswith("- ") or lines[j].startswith("  ")):
                if lines[j].lstrip().startswith("- "):
                    acc.append(lines[j])
                else:
                    acc[-1] += " " + lines[j].strip()
                j += 1
            yield "l", lists(acc); i = j; continue
        j = i; acc = []
        while j < len(lines) and lines[j].strip() and not lines[j].startswith(("#", "|")) and not lines[j].lstrip().startswith("- ") and not re.match(r"\d+\. ", lines[j]):
            acc.append(lines[j].strip()); j += 1
        cap = ' class="cap"' if acc[0].startswith("**Table") else ""; yield "p", f"<p{cap}>{inline(' '.join(acc))}</p>"; i = j


md = open(SRC, encoding="utf-8").read()
if FDIR == "figs2":                                           # placeholders from numbers.csv, includes, [A6|A7] marks
    for _ in range(2):
        md = re.sub(r"\{\{include:([^}]+)\}\}", lambda m: open(REP + m.group(1), encoding="utf-8").read(), md)
    NUM = {r["key"]: r["value"] for r in csv.DictReader(open(REP + "numbers.csv", encoding="utf-8"))}
    md = re.sub(r"\{\{(\w+)\}\}", lambda m: NUM[m.group(1)], md).replace("[A6|A7] ", "").replace(" [A6\\|A7] ", " ")
    assert "{{" not in md and "[A6" not in md
cut = md.index("\n## ", md.index("\n"))                       # cover = the title and its bullet block
cover, body = md[:cut], md[cut:]
title = inline(cover.splitlines()[0].lstrip("# ").strip())
meta = "".join(f"<div>{inline(l.strip()[2:])}</div>" for l in cover.splitlines()[1:] if l.strip().startswith("- "))
done, parts = set(), []
for kind, h in blocks(body):
    parts.append(h)
    txt = re.sub(r"<[^>]+>", "", h)
    for n in sorted(FIG):
        if n not in done and re.search(rf"(Fig\.|図)\s?{n}(?!\d)", txt):
            done.add(n); cls = "wide" if n in WIDE else ""
            parts.append(f'<figure class="{cls}"><img src="../{FDIR}/{FIG[n]}"><figcaption><b>{"図 " if JA and FDIR == "figs2" else "Fig. "}{n}.</b> {inline(CAP.get(n, ""))}</figcaption></figure>')
font = '"Yu Mincho", "MS Mincho", serif' if JA else '"Times New Roman", Times, serif'
css = f"""@page {{ size: Letter; margin: 0.7in 0.6in 0.75in 0.6in; }}
body {{ font-family: {font}; font-size: 9.6pt; line-height: 1.22; color: #000; margin: 0; }}
.title {{ column-span: all; text-align: center; margin-bottom: 10pt; }}
.title h1 {{ font-size: 17pt; font-weight: normal; margin: 0 0 6pt; line-height: 1.15; }}
.title .meta {{ font-size: 8.6pt; line-height: 1.3; }}
main {{ column-count: 2; column-gap: 0.24in; text-align: justify; hyphens: auto; }}
h2 {{ font-size: 10pt; font-variant: small-caps; letter-spacing: 0.3pt; margin: 9pt 0 3pt; break-after: avoid; }}
h3 {{ font-size: 9.6pt; font-style: italic; font-weight: normal; margin: 6pt 0 2pt; break-after: avoid; }}
h4 {{ font-size: 9.6pt; font-style: italic; font-weight: normal; margin: 5pt 0 1pt; break-after: avoid; }}
p {{ margin: 0 0 4pt; }} p.cap {{ break-after: avoid; font-size: 8.2pt; margin: 5pt 0 2pt; }} ul, ol {{ margin: 0 0 4pt; padding-left: 13pt; }} li {{ margin: 0 0 1.5pt; }}
code {{ font-family: Consolas, monospace; font-size: 8pt; }}
table {{ width: 100%; border-collapse: collapse; font-size: 7.4pt; line-height: 1.18; margin: 4pt 0 6pt; break-inside: avoid; text-align: left; }}
table.wide {{ column-span: all; }}
th {{ border-top: 0.8pt solid #000; border-bottom: 0.5pt solid #000; font-weight: bold; padding: 1.5pt 3pt; vertical-align: bottom; }}
td {{ border-bottom: 0.25pt solid #bbb; padding: 1.5pt 3pt; vertical-align: top; }}
figure {{ margin: 5pt 0 7pt; break-inside: avoid; }} figure.wide {{ column-span: all; }}
figure img {{ width: 100%; }} figcaption {{ font-size: 7.8pt; line-height: 1.2; margin-top: 2pt; }}"""
page = (f'<!doctype html><html lang="{"ja" if JA else "en"}"><head><meta charset="utf-8"><title>{title}</title><style>{css}</style></head>'
        f'<body><main><div class="title"><h1>{title}</h1><div class="meta">{meta}</div></div>{"".join(parts)}</main></body></html>')
os.makedirs(REP + "build", exist_ok=True); H = REP + "build/" + os.path.basename(OUT).replace(".pdf", ".html")
open(H, "w", encoding="utf-8").write(page)
prof = os.path.join(os.environ.get("TEMP", "C:/Windows/Temp"), "tfb_edge_profile")
subprocess.run([EDGE, "--headless=new", "--disable-gpu", "--no-pdf-header-footer", f"--user-data-dir={prof}", f"--print-to-pdf={OUT}",
                "file:///" + H.replace("\\", "/")], check=True, capture_output=True, timeout=180)
print(f"{OUT}: {len(PdfReader(OUT).pages)} pages | figures placed {sorted(done)} of {sorted(FIG)} | html {H}")
