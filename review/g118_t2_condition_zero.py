"""G118 (10/29, ledger #71, strategy J12): finA6 with the March (validation) queue predictions of one condition set to 0,
to read the public T2 per condition: on0 = the 40 March onset windows empty, og0 = the 40 March ongoing windows empty.
Text-level edit of finA6_final.zip's submission.csv: every other row stays byte-identical (April, Task 1 / 3 / 4); the
script checks that only the intended rows differ. Not for the finals.
usage: py g118_t2_condition_zero.py"""
import glob, io, zipfile, hashlib, pandas as pd

W = r"C:/Users/user/tfb/work/subs/"; REL = r"C:/Users/user/tfb/data/kaggle_public/"
RAW = zipfile.ZipFile(W + "finA6_final.zip").read("submission.csv").decode(); NL = "\r\n" if "\r\n" in RAW else "\n"
base = pd.read_csv(io.StringIO(RAW))
key = pd.read_csv(REL + "submission_key.csv", usecols=["submission_id", "task", "split", "window_id"], dtype=str)
cond = pd.concat([pd.read_csv(f, usecols=["window_id", "condition"]) for f in glob.glob(REL + "task2/*/validation/window_index.csv")])
key["cond"] = key.window_id.map(dict(zip(cond.window_id, cond.condition)))
assert (base.submission_id.astype(str).to_numpy() == key.submission_id.to_numpy()).all()
for tag, c in (("on0", "queue_onset"), ("og0", "queue_ongoing")):
    sel = ((key.task == "queue") & (key.split == "validation") & (key.cond == c)).to_numpy()
    lines = RAW.split(NL); n = 0
    for i in sel.nonzero()[0]:
        c_ = lines[i + 1].split(","); assert c_[1] == "queue"
        if c_[4] != "0":
            c_[4] = "0"; lines[i + 1] = ",".join(c_); n += 1
    txt = NL.join(lines); out = pd.read_csv(io.StringIO(txt))
    diff = (out.queue_pred != base.queue_pred).to_numpy(); assert not (diff & ~sel).any() and (out.queue_pred[sel] == 0).all()
    same = sum(a == b for a, b in zip(lines, RAW.split(NL)))
    print(tag, "rows set to 0:", int(sel.sum()), "| rows that changed:", int(diff.sum()), n, "| windows:", key.window_id[sel].nunique(),
          "| byte-identical lines:", same, "of", len(lines))
    z = W + f"finA6_{tag}.zip"
    with zipfile.ZipFile(z, "w", zipfile.ZIP_DEFLATED) as f:
        f.writestr("submission.csv", txt)
    print(z, hashlib.sha256(zipfile.ZipFile(z).read("submission.csv")).hexdigest()[:16])
