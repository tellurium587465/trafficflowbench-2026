"""A20 (10/23 night, strategy's audit day 1 (1), catastrophic risk): integrity of the final submission zips against the
release's sample_submission.csv and README rules ("one row for every value the leaderboard scores ... state wants
speed_kmh and flow_vph, queue wants queue_pred as 0 or 1, odme wants path_flow. Leave the rest at 0. Never leave a cell
empty"). Per zip: exactly one member submission.csv; the same columns; the same submission_id sequence and task per id as
the sample; no duplicate id; no blank / non-finite value; queue_pred in {0, 1}; speed_kmh, flow_vph, path_flow >= 0;
the columns a task does not ask for are 0. Across the zips: which tasks differ (state / odme must be identical between A,
B and s80q; queue only on the private month).
usage: python a20_zip_integrity.py <zip> [<zip> ...]   (names under work/subs)"""
import sys, zipfile, numpy as np, pandas as pd

REL = "C:/Users/user/tfb/data/kaggle_public/"; SUBS = "C:/Users/user/tfb/work/subs/"
S = pd.read_csv(REL + "sample_submission.csv"); K = pd.read_csv(REL + "submission_key.csv", usecols=["submission_id", "split"])
COLS = list(S.columns); ASK = {"state": ["speed_kmh", "flow_vph"], "queue": ["queue_pred"], "odme": ["path_flow"]}
print(f"sample: {len(S)} rows, tasks {S.task.value_counts().to_dict()}", flush=True)
subs, ok_all = {}, True
for name in sys.argv[1:]:
    z = zipfile.ZipFile(SUBS + name); mem = [i.filename for i in z.infolist()]; ok = mem == ["submission.csv"]
    D = pd.read_csv(z.open("submission.csv"), keep_default_na=False, dtype=str); subs[name] = D
    chk = {"one member submission.csv": ok, "columns": list(D.columns) == COLS, "rows": len(D) == len(S)}
    chk["id sequence"] = chk["rows"] and (D.submission_id.astype(np.int64).values == S.submission_id.values).all()
    chk["no duplicate id"] = not D.submission_id.duplicated().any()
    chk["task per id"] = chk["rows"] and (D.task.values == S.task.values).all()
    chk["no blank"] = not (D[COLS[2:]] == "").any().any()
    N = D[COLS[2:]].apply(pd.to_numeric, errors="coerce")
    chk["all numeric and finite"] = bool(np.isfinite(N.values).all())
    q = N.queue_pred[D.task == "queue"]; chk["queue_pred in {0,1}"] = bool(q.isin([0, 1]).all())
    chk["state >= 0"] = bool((N.loc[D.task == "state", ["speed_kmh", "flow_vph"]] >= 0).all().all())
    chk["path_flow >= 0"] = bool((N.path_flow[D.task == "odme"] >= 0).all())
    chk["unasked columns 0"] = all(bool((N.loc[D.task == t, [c for c in COLS[2:] if c not in ASK[t]]] == 0).all().all()) for t in ASK)
    ok_all &= all(chk.values())
    print(f"{name}: {'PASS' if all(chk.values()) else 'FAIL'} | " + ", ".join(f"{k} {v}" for k, v in chk.items()), flush=True)
    print(f"   queue positives {int(q.sum())} of {len(q)}; state speed range {N.speed_kmh[D.task == 'state'].min():.1f}-"
          f"{N.speed_kmh[D.task == 'state'].max():.1f}; path_flow sum {N.path_flow[D.task == 'odme'].sum():.1f}", flush=True)
names = list(subs)
for i in range(len(names)):
    for j in range(i + 1, len(names)):
        a, b = subs[names[i]], subs[names[j]]; diff = (a[COLS[2:]] != b[COLS[2:]]).any(axis=1)
        by = pd.DataFrame({"task": a.task[diff], "split": K.split[diff.values]}).value_counts().to_dict()
        print(f"{names[i]} vs {names[j]}: rows differing {int(diff.sum())} by (task, split) {by}")
print("A20 zip integrity:", "PASS" if ok_all else "FAIL")
