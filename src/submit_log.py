"""Submit to Kaggle and record provenance (harness review: lineage was not on disk).

Appends one JSON line per submission to work/subs/SUBMISSIONS.jsonl: time, zip, message, git commit (+ dirty
flag), the component files (T1 state / T2 queue / T4 odme) with size and sha1, and the zip's sha1. Refuses to
submit if a component file is missing.

usage: python submit_log.py <zip_name_in_work/subs> "<message>" <t1_state> <t2_queue> <t4_odme> [--dry]
"""
import hashlib, json, os, subprocess, sys, time

W = "C:/Users/user/tfb/"
KAGGLE = r"C:\Users\user\AppData\Local\Python\bin\kaggle.exe"
COMP = "2026-ieee-big-data-traffic-flow-bench"


def sha1(path):
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def git(*a):
    return subprocess.run(["git", "-C", W, *a], capture_output=True, text=True, encoding="utf-8", errors="replace").stdout.strip()


zip_name, msg, *parts = sys.argv[1:]
dry = "--dry" in parts; parts = [p for p in parts if p != "--dry"]
zpath = W + "work/subs/" + zip_name
comps = dict(zip(("t1_state", "t2_queue", "t4_odme"), parts))
for k, p in comps.items():
    if not os.path.exists(p):
        sys.exit(f"missing component {k}: {p}")
rec = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "zip": zip_name, "zip_sha1": sha1(zpath), "message": msg,
       "git_commit": git("rev-parse", "--short", "HEAD"), "git_dirty": bool(git("status", "--porcelain", "--", "src")),
       "components": {k: {"path": p, "bytes": os.path.getsize(p), "sha1": sha1(p)} for k, p in comps.items()}}
if not dry:
    r = subprocess.run([KAGGLE, "competitions", "submit", "-c", COMP, "-f", zpath, "-m", msg], capture_output=True, text=True, encoding="utf-8", errors="replace")
    rec["kaggle_stdout"] = (r.stdout or "").strip().splitlines()[-2:]
    if r.returncode != 0:
        rec["kaggle_error"] = (r.stderr or "")[-500:]
with open(W + "work/subs/SUBMISSIONS.jsonl", "a", encoding="utf-8") as f:
    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
print(json.dumps(rec, ensure_ascii=False, indent=1))
